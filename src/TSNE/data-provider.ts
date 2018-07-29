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

import { ColumnStats, DataPoint, DataSet, PointMetadata, SpriteAndMetadataInfo } from './data';
import * as util from './util';

/** Maximum number of colors supported in the color map. */
const NUM_COLORS_COLOR_MAP = 50;
const MAX_SPRITE_IMAGE_SIZE_PX = 8192;

export const METADATA_MSG_ID = 'metadata';
export const TENSORS_MSG_ID = 'tensors';

/** Matches the json format of `projector_config.proto` */
export interface SpriteMetadata {
    imagePath: string;
    singleImageDim: [number, number];
}

/** Matches the json format of `projector_config.proto` */
export interface EmbeddingInfo {
    /** Name of the tensor. */
    tensorName: string;
    /** The shape of the tensor. */
    tensorShape: [number, number];
    /**
     * The path to the tensors TSV file. If empty, it is assumed that the tensor
     * is stored in the checkpoint file.
     */
    tensorPath?: string;
    /** The path to the metadata file associated with the tensor. */
    metadataPath?: string;
    /** The path to the bookmarks file associated with the tensor. */
    bookmarksPath?: string;
    sprite?: SpriteMetadata;
}

/**
 * Matches the json format of `projector_config.proto`
 * This should be kept in sync with the code in vz-projector-data-panel which
 * holds a template for users to build a projector config JSON object from the
 * projector UI.
 */
export interface ProjectorConfig {
    embeddings: EmbeddingInfo[];
    modelCheckpointPath?: string;
}

export type ServingMode = 'demo' | 'server' | 'proto';

/** Interface between the data storage and the UI. */
export interface DataProvider {
    /** Returns a list of run names that have embedding config files. */
    retrieveRuns(callback: (runs: string[]) => void): void;

    /**
     * Returns the projector configuration: number of tensors, their shapes,
     * and their associated metadata files.
     */
    retrieveProjectorConfig(run: string, callback: (d: ProjectorConfig) => void): void;

    /** Fetches and returns the tensor with the specified name. */
    retrieveTensor(run: string, tensorName: string, callback: (ds: DataSet) => void): any;

    /**
     * Fetches the metadata for the specified tensor.
     */
    retrieveSpriteAndMetadata(run: string, tensorName: string, callback: (r: SpriteAndMetadataInfo) => void): void;

    getBookmarks(run: string, tensorName: string, callback: (r: any[]) => void):
        void;
}

export function parseRawTensors(content: ArrayBuffer, callback: (ds: DataSet) => void) {
    parseTensors(content).then((data) => {
        callback(new DataSet(data));
    });
}

export function parseRawMetadata(contents: ArrayBuffer, callback: (r: SpriteAndMetadataInfo) => void) {
    parseMetadata(contents).then((result) => callback(result));
}

/**
 * Parse an ArrayBuffer in a streaming fashion line by line (or custom delim).
 * Can handle very large files.
 *
 * @param content The array buffer.
 * @param callback The callback called on each line.
 * @param chunkSize The size of each read chunk, defaults to ~1MB. (optional)
 * @param delim The delimiter used to split a line, defaults to '\n'. (optional)
 * @returns A promise for when it is finished.
 */
function streamParse(content: ArrayBuffer, callback: (line: string) => void, chunkSize = 1000000, delim = '\n'): Promise<void> {

    return new Promise<void>((resolve, reject) => {
        let offset = 0;
        const bufferSize = content.byteLength - 1;
        let data = '';

        function readHandler(str: any) {
            offset += chunkSize;
            const parts = str.split(delim);
            const first = data + parts[0];
            if (parts.length === 1) {
                data = first;
                readChunk(offset, chunkSize);
                return;
            }
            data = parts[parts.length - 1];
            callback(first);
            for (let i = 1; i < parts.length - 1; i++) {
                callback(parts[i]);
            }
            if (offset >= bufferSize) {
                if (data) {
                    callback(data);
                }
                resolve();
                return;
            }
            readChunk(offset, chunkSize);
        }

        function readChunk(offset2: number, size: number) {
            const contentChunk = content.slice(offset2, offset2 + size);

            const blob = new Blob([contentChunk]);
            const file = new FileReader();
            file.onload = (e: any) => readHandler(e.target.result);
            file.readAsText(blob);
        }

        readChunk(offset, chunkSize);
    });
}

/** Parses a tsv text file. */
export function parseTensors(
    content: ArrayBuffer, valueDelim = '\t'): Promise<DataPoint[]> {
    // logging.setModalMessage('Parsing tensors...', TENSORS_MSG_ID);

    console.log('content:', content);

    return new Promise<DataPoint[]>((resolve) => {
        
        const data: DataPoint[] = [];
        let numDim: number;

        streamParse(content, (line: string) => {

            console.log('line:', valueDelim);

            line = line.trim();
            if (line === '') {
                return;
            }
            const row = line.split(valueDelim);
            const dataPoint: DataPoint = {
                index: data.length,
                metadata: {},
                projections: {},
                vector: new Float32Array(0)
            };
            // If the first label is not a number, take it as the label.
            if (isNaN(row[0] as any) || numDim === row.length - 1) {
                dataPoint.metadata['label'] = row[0];
                dataPoint.vector = new Float32Array(row.slice(1).map(Number));
            } else {
                dataPoint.vector = new Float32Array(row.map(Number));
            }
            data.push(dataPoint);
            if (numDim == null) {
            numDim = dataPoint.vector.length;
            }
            if (numDim !== dataPoint.vector.length) {
                // logging.setModalMessage('Parsing failed. Vector dimensions do not match');
                throw Error('Parsing failed. Vector dimensions do not match');
            }
            if (numDim <= 1) {
                // logging.setModalMessage('Parsing failed. Found a vector with only one dimension?');
                throw Error('Parsing failed. Found a vector with only one dimension?');
            }
        }).then(() => {
            // logging.setModalMessage(null, TENSORS_MSG_ID);
            resolve(data);
        });
    });
}

/** Parses a tsv text file. */
export function parseTensorsFromFloat32Array(data: Float32Array, dim: number): Promise<DataPoint[]> {
    return util.runAsyncTask('Parsing tensors...', () => {

        const N = data.length / dim;
        const dataPoints: DataPoint[] = [];
        let offset = 0;
        for (let i = 0; i < N; ++i) {
            dataPoints.push({
                index: i,
                metadata: {},
                projections: {},
                vector: data.subarray(offset, offset + dim),
            });
            offset += dim;
        }

        return dataPoints;

    }, TENSORS_MSG_ID).then((dataPoints) => {
        // logging.setModalMessage(null, TENSORS_MSG_ID);
        return dataPoints;
    });
}

export function analyzeMetadata(columnNames: any, pointsMetadata: PointMetadata[]): ColumnStats[] {

    const columnStats: ColumnStats[] = columnNames.map((name: any) => {
        return {
            isNumeric: true,
            max: Number.NEGATIVE_INFINITY,
            min: Number.POSITIVE_INFINITY,
            name,
            tooManyUniqueValues: false
        };
    });

    const mapOfValues: [{[value: string]: number}] =
        columnNames.map(() => new Object());

    pointsMetadata.forEach((metadata: any) => {
        columnNames.forEach((name: string, colIndex: number) => {
            const stats = columnStats[colIndex];
            const map = mapOfValues[colIndex];
            const value = metadata[name];

            // Skip missing values.
            if (value == null) {
                return;
            }

            if (!stats.tooManyUniqueValues) {
                if (value in map) {
                    map[value]++;
                } else {
                    map[value] = 1;
                }
                if (Object.keys(map).length > NUM_COLORS_COLOR_MAP) {
                    stats.tooManyUniqueValues = true;
                }
            }
            if (isNaN(value as any)) {
                stats.isNumeric = false;
            } else {
                metadata[name] = +value;
                stats.min = Math.min(stats.min, +value);
                stats.max = Math.max(stats.max, +value);
            }
        });
    });

    columnStats.forEach((stats, colIndex) => {
        stats.uniqueEntries = Object.keys(mapOfValues[colIndex]).map((label: any) => {
            return {label, count: mapOfValues[colIndex][label]};
        });
    });

    return columnStats;
}

export function parseMetadata(content: ArrayBuffer): Promise<SpriteAndMetadataInfo> {
    // logging.setModalMessage('Parsing metadata...', METADATA_MSG_ID);

    return new Promise<SpriteAndMetadataInfo>((resolve, reject) => {
        const pointsMetadata: PointMetadata[] = [];
        let hasHeader = false;
        let lineNumber = 0;
        let columnNames = ['label'];
        streamParse(content, (line: string) => {
            if (line.trim().length === 0) {
                return;
            }
            if (lineNumber === 0) {
                hasHeader = line.indexOf('\t') >= 0;

                // If the first row doesn't contain metadata keys, we assume that the
                // values are labels.
                if (hasHeader) {
                    columnNames = line.split('\t');
                    lineNumber++;
                    return;
                }
            }

            lineNumber++;

            const rowValues = line.split('\t');
            const metadata: PointMetadata = {};
            pointsMetadata.push(metadata);
            columnNames.forEach((name: string, colIndex: number) => {
                let value: any = rowValues[colIndex];
                // Normalize missing values.
                value = (value === '' ? null : value);
                metadata[name] = value;
            });

        }).then(() => {
            // logging.setModalMessage(null, METADATA_MSG_ID);
            resolve({
                pointsInfo: pointsMetadata,
                stats: analyzeMetadata(columnNames, pointsMetadata)
            });
        });
    });
}

export function fetchImage(url: string): Promise<HTMLImageElement> {
    return new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = (err) => reject(err);
        image.crossOrigin = '';
        image.src = url;
    });
}

export function retrieveSpriteAndMetadataInfo(metadataPath: string, spriteImagePath: string, spriteMetadata: SpriteMetadata, callback: (r: SpriteAndMetadataInfo) => void) {
    let metadataPromise: Promise<SpriteAndMetadataInfo> = Promise.resolve({});
    if (metadataPath) {
        metadataPromise = new Promise<SpriteAndMetadataInfo>((resolve, reject) => {
        // logging.setModalMessage('Fetching metadata...', METADATA_MSG_ID);

            const request = new XMLHttpRequest();
            request.open('GET', metadataPath);
            request.responseType = 'arraybuffer';

            request.onreadystatechange = () => {
            if (request.readyState === 4) {
                if (request.status === 200) {
                // The metadata was successfully retrieved. Parse it.
                resolve(parseMetadata(request.response));
                } else {
                // The response contains the error message, but we must convert it
                // to a string.
                const errorReader = new FileReader();
                errorReader.onload = () => {
                    // logging.setErrorMessage(errorReader.result, 'fetching metadata');
                    reject();
                };
                errorReader.readAsText(new Blob([request.response]));
                }
            }
            };
            request.send(null);
        });
    }
    let spriteMsgId: any = null;
    let spritesPromise: Promise<HTMLImageElement> = Promise.reject(null);
    if (spriteImagePath) {
        // spriteMsgId = logging.setModalMessage('Fetching sprite image...');
        spriteMsgId = 'awesome';
        spritesPromise = fetchImage(spriteImagePath);
    }

    // Fetch the metadata and the image in parallel.
    Promise.all([metadataPromise, spritesPromise]).then((values: any) => {
        if (spriteMsgId) {
            // logging.setModalMessage(null, spriteMsgId);
        }
        const [metadata, spriteImage] = values;

        if (spriteImage && (spriteImage.height > MAX_SPRITE_IMAGE_SIZE_PX || spriteImage.width > MAX_SPRITE_IMAGE_SIZE_PX)) {
            // logging.setModalMessage(`Error: Sprite image of dimensions ${spriteImage.width}px x ` + `${spriteImage.height}px exceeds maximum dimensions ` + `${MAX_SPRITE_IMAGE_SIZE_PX}px x ${MAX_SPRITE_IMAGE_SIZE_PX}px`);
        } else {
            metadata.spriteImage = spriteImage;
            metadata.spriteMetadata = spriteMetadata;
            try {
                callback(metadata);
            } catch (e) {
                // logging.setModalMessage(String(e));
            }
        }
    });
}