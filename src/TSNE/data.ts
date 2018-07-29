/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import { SpriteMetadata } from './data-provider';
import * as util from './util';
import { centroid, cosDistNorm, dot, norm2, sub, unit, Vector } from './vector';
import { TSNE } from './bh_tsne';
import { NearestEntry, findKNNofPoint, findKNNGPUCosine, findKNN } from './knn';

export type DistanceFunction = (a: Vector, b: Vector) => number;
export type ProjectionComponents3D = [string, string, string];

export interface PointMetadata { [key: string]: number|string; }

export interface DataProto {
    shape: [number, number];
    tensor: number[];
    metadata: {
        columns: Array<{name: string; stringValues: string[]; numericValues: number[]; } >;
        sprite: {imageBase64: string; singleImageDim: [number, number]};
    };
}

/** Statistics for a metadata column. */
export interface ColumnStats {
    name: string;
    isNumeric: boolean;
    tooManyUniqueValues: boolean;
    uniqueEntries?: Array<{label: string, count: number}>;
    min: number;
    max: number;
}

export interface SpriteAndMetadataInfo {
    stats?: ColumnStats[];
    pointsInfo?: PointMetadata[];
    spriteImage?: HTMLImageElement;
    spriteMetadata?: SpriteMetadata;
}

/** A single collection of points which make up a sequence through space. */
export interface Sequence {
    /** Indices into the DataPoints array in the Data object. */
    pointIndices: number[];
}

export interface DataPoint {
    /** The point in the original space. */
    vector: Float32Array;

    /*
    * Metadata for each point. Each metadata is a set of key/value pairs
    * where the value can be a string or a number.
    */
    metadata: PointMetadata;

    /** index of the sequence, used for highlighting on click */
    sequenceIndex?: number;

    /** index in the original data source */
    index: number;

    /** This is where the calculated projections space are cached */
    projections: {[key: string]: number};
}

const IS_FIREFOX = navigator.userAgent.toLowerCase().indexOf('firefox') >= 0;
/** Controls whether nearest neighbors computation is done on the GPU or CPU. */
const KNN_GPU_ENABLED = util.hasWebGLSupport() && !IS_FIREFOX;

export const TSNE_SAMPLE_SIZE = 10000;

/**
 * Reserved metadata attributes used for sequence information
 * NOTE: Use "__seq_next__" as "__next__" is deprecated.
 */
const SEQUENCE_METADATA_ATTRS = ['__next__', '__seq_next__'];

function getSequenceNextPointIndex(pointMetadata: PointMetadata): number|null {
    let sequenceAttr = null;
    for (const metadataAttr of SEQUENCE_METADATA_ATTRS) {
        if (metadataAttr in pointMetadata && pointMetadata[metadataAttr] !== '') {
            sequenceAttr = pointMetadata[metadataAttr];
            break;
        }
    }
    if (sequenceAttr == null) {
        return null;
    }
    return +sequenceAttr;
}

/**
 * Dataset contains a DataPoints array that should be treated as immutable. This
 * acts as a working subset of the original data, with cached properties
 * from computationally expensive operations. Because creating a subset
 * requires normalizing and shifting the vector space, we make a copy of the
 * data so we can still always create new subsets based on the original data.
 */
export class DataSet {
    public points: DataPoint[];
    public sequences: Sequence[];

    public shuffledDataIndices: number[] = [];

    /**
     * This keeps a list of all current projections so you can easily test to see
     * if it's been calculated already.
     */
    public projections: {[projection: string]: boolean} = {};
    public nearest: NearestEntry[][] = [];
    public nearestK: number = 0;
    public tSNEIteration: number = 0;
    public tSNEShouldPause = false;
    public tSNEShouldStop = true;
    public superviseFactor: number = 0;
    public superviseLabels: string[] = [];
    public superviseInput: string = '';
    public dim: [number, number] = [0, 0];
    public hasTSNERun: boolean = false;
    public spriteAndMetadataInfo: SpriteAndMetadataInfo|undefined;
    public fracVariancesExplained: number[] = [];

    private tsne: any;

    /** Creates a new Dataset */
    constructor(
        points: DataPoint[], spriteAndMetadataInfo?: SpriteAndMetadataInfo) {
        this.points = points;
        this.shuffledDataIndices = util.shuffle(util.range(this.points.length));
        this.sequences = this.computeSequences(points);
        this.dim = [this.points.length, this.points[0].vector.length];
        this.spriteAndMetadataInfo = spriteAndMetadataInfo;
    }

    private computeSequences(points: DataPoint[]) {
        // Keep a list of indices seen so we don't compute sequences for a given
        // point twice.
        const indicesSeen = new Int8Array(points.length);
        // Compute sequences.
        const indexToSequence: {[index: number]: Sequence} = {};
        const sequences: Sequence[] = [];
        for (let i = 0; i < points.length; i++) {
            if (indicesSeen[i]) {
                continue;
            }
            indicesSeen[i] = 1;

            // Ignore points without a sequence attribute.
            let next = getSequenceNextPointIndex(points[i].metadata);
            if (next == null) {
                continue;
            }
            if (next in indexToSequence) {
                const existingSequence = indexToSequence[next];
                // Pushing at the beginning of the array.
                existingSequence.pointIndices.unshift(i);
                indexToSequence[i] = existingSequence;
                continue;
            }
            // The current point is pointing to a new/unseen sequence.
            const newSequence: Sequence = {pointIndices: []};
            indexToSequence[i] = newSequence;
            sequences.push(newSequence);
            let currentIndex = i;
            while (points[currentIndex]) {
                newSequence.pointIndices.push(currentIndex);
                next = getSequenceNextPointIndex(points[currentIndex].metadata);
                if (next != null) {
                    indicesSeen[next] = 1;
                    currentIndex = next;
                } else {
                    currentIndex = -1;
                }
            }
        }
        return sequences;
    }

    public projectionCanBeRendered(projection: ProjectionType): boolean {
        if (projection !== 'tsne') {
            return true;
        }
        return this.tSNEIteration > 0;
    }

    /**
     * Returns a new subset dataset by copying out data. We make a copy because
     * we have to modify the vectors by normalizing them.
     *
     * @param subset Array of indices of points that we want in the subset.
     *
     * @return A subset of the original dataset.
     */
    public getSubset(subset?: number[]): DataSet {
        const pointsSubset = ((subset != null) && (subset.length > 0)) ? subset.map((i: number) => this.points[i]) : this.points;
        const points = pointsSubset.map((dp: any) => {
            return {
                metadata: dp.metadata,
                index: dp.index,
                vector: dp.vector.slice(),
                projections: {} as {[key: string]: number}
            };
        });
        return new DataSet(points, this.spriteAndMetadataInfo);
    }

    /**
     * Computes the centroid, shifts all points to that centroid,
     * then makes them all unit norm.
     */
    public normalize() {
        // Compute the centroid of all data points.
        const centroidVal = centroid(this.points, (a) => a.vector);
        if (centroidVal == null) {
            throw Error('centroid should not be null');
        }
        // Shift all points by the centroid and make them unit norm.
        for (let id = 0; id < this.points.length; ++id) {
            const dataPoint = this.points[id];
            dataPoint.vector = sub(dataPoint.vector, centroidVal);
            if (norm2(dataPoint.vector) > 0) {
            // If we take the unit norm of a vector of all 0s, we get a vector of
            // all NaNs. We prevent that with a guard.
                unit(dataPoint.vector);
            }
        }
    }

    /** Projects the dataset onto a given vector and caches the result. */
    public projectLinear(dir: Vector, label: string) {
    this.projections[label] = true;
    this.points.forEach((dataPoint) => {
        dataPoint.projections[label] = dot(dataPoint.vector, dir);
    });
    }

    /** Runs tsne on the data. */
    public projectTSNE(perplexity: number, learningRate: number, tsneDim: number, stepCallback: (iter: number|null) => void) {

        this.hasTSNERun = true;
        const k = Math.floor(3 * perplexity);
        const opt = {epsilon: learningRate, perplexity, dim: tsneDim};
        this.tsne = new TSNE(opt);
        this.tsne.setSupervision(this.superviseLabels, this.superviseInput);
        this.tsne.setSuperviseFactor(this.superviseFactor);
        this.tSNEShouldPause = false;
        this.tSNEShouldStop = false;
        this.tSNEIteration = 0;

        const sampledIndices = this.shuffledDataIndices.slice(0, TSNE_SAMPLE_SIZE);

        const step = () => {

            if (this.tSNEShouldStop) {
                this.projections['tsne'] = false;
                stepCallback(null);
                this.tsne = null;
                this.hasTSNERun = false;
                return;
            }

            if (!this.tSNEShouldPause) {

                this.tsne.step();
                const result = this.tsne.getSolution();

                sampledIndices.forEach((index, i) => {
                    const dataPoint = this.points[index];

                    dataPoint.projections['tsne-0'] = result[i * tsneDim + 0];
                    dataPoint.projections['tsne-1'] = result[i * tsneDim + 1];
                    if (tsneDim === 3) {
                        dataPoint.projections['tsne-2'] = result[i * tsneDim + 2];
                    }
                });

                this.projections['tsne'] = true;
                this.tSNEIteration++;
                stepCallback(this.tSNEIteration);
            }

            requestAnimationFrame(step);
        };

        // Nearest neighbors calculations.
        let knnComputation: Promise<NearestEntry[][]>;

        if (this.nearest != null && k === this.nearestK) {
            // We found the nearest neighbors before and will reuse them.
            knnComputation = Promise.resolve(this.nearest);
        } else {
            const sampledData = sampledIndices.map((i) => this.points[i]);
            this.nearestK = k;
            knnComputation = KNN_GPU_ENABLED ?
                findKNNGPUCosine(sampledData, k, ((d) => d.vector)) :
                findKNN(sampledData, k, ((d: any) => d.vector), (a: any, b: any, limit: any) => cosDistNorm(a, b));
        }

        knnComputation.then((nearest) => {
            this.nearest = nearest;
            util.runAsyncTask('Initializing T-SNE...', () => {
                this.tsne.initDataDist(this.nearest);
            }).then(step);
        });
    }

        /* Perturb TSNE and update dataset point coordinates. */
    public perturbTsne() {
        if (this.hasTSNERun && this.tsne) {
            this.tsne.perturb();
            const tsneDim = this.tsne.getDim();
            const result = this.tsne.getSolution();
            const sampledIndices = this.shuffledDataIndices.slice(0, TSNE_SAMPLE_SIZE);

            sampledIndices.forEach((index, i) => {
                const dataPoint = this.points[index];

                dataPoint.projections['tsne-0'] = result[i * tsneDim + 0];
                dataPoint.projections['tsne-1'] = result[i * tsneDim + 1];
                if (tsneDim === 3) {
                    dataPoint.projections['tsne-2'] = result[i * tsneDim + 2];
                }
            });
        }
    }

    public setSupervision(superviseColumn: string, superviseInput?: string) {
        if (superviseColumn != null) {
            const sampledIndices = this.shuffledDataIndices.slice(0, TSNE_SAMPLE_SIZE);
            const labels = new Array(sampledIndices.length);
            sampledIndices.forEach((index, i) => {
                labels[i] = this.points[index].metadata[superviseColumn].toString();
            });
            this.superviseLabels = labels;
        }
        if (superviseInput != null) {
            this.superviseInput = superviseInput;
        }
        if (this.tsne) {
            this.tsne.setSupervision(this.superviseLabels, this.superviseInput);
        }
    }

    public setSuperviseFactor(superviseFactor: number) {
        if (superviseFactor != null) {
            this.superviseFactor = superviseFactor;
            if (this.tsne) {
                this.tsne.setSuperviseFactor(superviseFactor);
            }
        }
    }

    /**
     * Merges metadata to the dataset and returns whether it succeeded.
     */
    public mergeMetadata(metadata: SpriteAndMetadataInfo): boolean {
        if (metadata && metadata.pointsInfo && metadata.pointsInfo.length !== this.points.length) {

            // const errorMessage = `Number of tensors (${this.points.length}) do not` + ` match the number of lines in metadata` + ` (${metadata.pointsInfo.length}).`;

            if (metadata.stats && metadata.stats.length === 1 && this.points.length + 1 === metadata.pointsInfo.length) {

                // If there is only one column of metadata and the number of points is
                // exactly one less than the number of metadata lines, this is due to an
                // unnecessary header line in the metadata and we can show a meaningful
                // error.
                // logging.setErrorMessage(errorMessage + ' Single column metadata should not have a header ' + 'row.', 'merging metadata');
                return false;

            } else if (metadata.stats && metadata.stats.length > 1 && this.points.length - 1 === metadata.pointsInfo.length) {
                // If there are multiple columns of metadata and the number of points is
                // exactly one greater than the number of lines in the metadata, this
                // means there is a missing metadata header.
                // logging.setErrorMessage(errorMessage + ' Multi-column metadata should have a header ' + 'row with column labels.', 'merging metadata');
                return false;
            }

            // logging.setWarningMessage(errorMessage);
        }

        this.spriteAndMetadataInfo = metadata;

        if (metadata.pointsInfo) {
            metadata.pointsInfo.slice(0, this.points.length).forEach((m, i) => {
                this.points[i].metadata = m;
            });
            return true;
        }

        return false;
    }

    public stopTSNE() {
        this.tSNEShouldStop = true;
    }

    /**
     * Finds the nearest neighbors of the query point using a
     * user-specified distance metric.
     */
    public findNeighbors(pointIndex: number, distFunc: DistanceFunction, numNN: number): NearestEntry[] {
    // Find the nearest neighbors of a particular point.
        const neighbors = findKNNofPoint(this.points, pointIndex, numNN, ((d) => d.vector), distFunc);
        // TODO(@dsmilkov): Figure out why we slice.
        const result = neighbors.slice(0, numNN);
        return result;
    }

    /**
     * Search the dataset based on a metadata field.
     */
    public query(query: string, inRegexMode: boolean, fieldName: string): number[] {
        const predicate = util.getSearchPredicate(query, inRegexMode, fieldName);
        const matches: number[] = [];
        this.points.forEach((point, id) => {
            if (predicate(point)) {
            matches.push(id);
            }
        });
        return matches;
    }
}

export type ProjectionType = 'tsne' | 'pca' | 'custom';

export class Projection {
    constructor(
        public projectionType: ProjectionType,
        public projectionComponents: ProjectionComponents3D,
        public dimensionality: number, public dataSet: DataSet) {}
}

export interface ColorOption {
    name: string;
    desc?: string;
    map?: (value: string|number) => string;
    /** List of items for the color map. Defined only for categorical map. */
    items?: Array<{label: string, count: number}>;
    /** Threshold values and their colors. Defined for gradient color map. */
    thresholds?: Array<{value: number, color: string}>;
    isSeparator?: boolean;
    tooManyUniqueValues?: boolean;
}

/**
 * An interface that holds all the data for serializing the current state of
 * the world.
 */
export class State {
    /** A label identifying this state. */
    public label: string = '';

    /** Whether this State is selected in the bookmarks pane. */
    public isSelected: boolean = false;

    /** The selected projection tab. */
    public selectedProjection: any; // ProjectionType

    /** Dimensions of the DataSet. */
    public dataSetDimensions: [number, number] = [0, 0];

    /** t-SNE parameters */
    public tSNEIteration: number = 0;
    public tSNEPerplexity: number = 0;
    public tSNELearningRate: number = 0;
    public tSNEis3d: boolean = true;

    /** PCA projection component dimensions */
    public pcaComponentDimensions: number[] = [];

    /** Custom projection parameters */
    public customSelectedSearchByMetadataOption: string = '';
    public customXLeftText: string = '';
    public customXLeftRegex: boolean = false;
    public customXRightText: string = '';
    public customXRightRegex: boolean = false;
    public customYUpText: string = '';
    public customYUpRegex: boolean = false;
    public customYDownText: string = '';
    public customYDownRegex: boolean = false;

    /** The computed projections of the tensors. */
    public projections: Array<{[key: string]: number}> = [];

    /** Filtered dataset indices. */
    public filteredPoints: number[] = [];

    /** The indices of selected points. */
    public selectedPoints: number[] = [];

    /** Camera state (2d/3d, position, target, zoom, etc). */
    public cameraDef: any; // CamerDef

    /** Color by option. */
    public selectedColorOptionName: string = '';
    public forceCategoricalColoring: boolean = false;

    /** Label by option. */
    public selectedLabelOption: string = '';
}

export function getProjectionComponents(projection: ProjectionType, components: Array<(number|string)>): ProjectionComponents3D {
    if (components.length > 3) {
        throw new RangeError('components length must be <= 3');
    }
    const projectionComponents: [any, any, any] = [null, null, null];
    const prefix = (projection === 'custom') ? 'linear' : projection;
    for (let i = 0; i < components.length; ++i) {
        if (components[i] == null) {
            continue;
        }
        projectionComponents[i] = prefix + '-' + components[i];
    }
    return projectionComponents;
}

export function stateGetAccessorDimensions(state: State): Array<number|string> {
    let dimensions: Array<number|string>;
    switch (state.selectedProjection) {
    case 'pca':
        dimensions = state.pcaComponentDimensions.slice();
        break;
    case 'tsne':
        dimensions = [0, 1];
        if (state.tSNEis3d) {
        dimensions.push(2);
        }
        break;
    case 'custom':
        dimensions = ['x', 'y'];
        break;
    default:
        throw new Error('Unexpected fallthrough');
    }
    return dimensions;
}