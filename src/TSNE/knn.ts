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
import * as util from './util.js';
import { toTypedArray, cosDistNorm, Vector } from './vector.js';
import { KMin } from './heap.js';

export interface NearestEntry {
    index: number;
    dist: number;
}

/**
 * Optimal size for the height of the matrix when doing computation on the GPU
 * using WebGL. This was found experimentally.
 *
 * This also guarantees that for computing pair-wise distance for up to 10K
 * vectors, no more than 40MB will be allocated in the GPU. Without the
 * allocation limit, we can freeze the graphics of the whole OS.
 */
const OPTIMAL_GPU_BLOCK_SIZE = 256;
/** Id of message box used for knn gpu progress bar. */
const KNN_GPU_MSG_ID = 'knn-gpu';

/**
 * Returns the K nearest neighbors for each vector where the distance
 * computation is done on the GPU (WebGL) using cosine distance.
 *
 * @param dataPoints List of data points, where each data point holds an
 *   n-dimensional vector.
 * @param k Number of nearest neighbors to find.
 * @param accessor A method that returns the vector, given the data point.
 */
export function findKNNGPUCosine<T>(dataPoints: T[], k: number, accessor: (dataPoint: T) => Float32Array): Promise<NearestEntry[][]> {
    const N = dataPoints.length;
    const dim = accessor(dataPoints[0]).length;

    // The goal is to compute a large matrix multiplication A*A.T where A is of
    // size NxD and A.T is its transpose. This results in a NxN matrix which
    // could be too big to store on the GPU memory. To avoid memory overflow, we
    // compute multiple A*partial_A.T where partial_A is of size BxD (B is much
    // smaller than N). This results in storing only NxB size matrices on the GPU
    // at a given time.

    // A*A.T will give us NxN matrix holding the cosine distance between every
    // pair of points, which we sort using KMin data structure to obtain the
    // K nearest neighbors for each point.
    const typedArray = toTypedArray(dataPoints, accessor);
    const bigMatrix = new weblas.pipeline.Tensor([N, dim], typedArray);
    const nearest: NearestEntry[][] = new Array(N);
    const numPieces = Math.ceil(N / OPTIMAL_GPU_BLOCK_SIZE);
    const M = Math.floor(N / numPieces);
    const modulo = N % numPieces;
    let offset = 0;
    let progress = 0;
    const progressDiff = 1 / (2 * numPieces);
    let piece = 0;

    function step(resolve: (result: NearestEntry[][]) => void) {
        const progressMsg = 'Finding nearest neighbors: ' + (progress * 100).toFixed() + '%';
        util.runAsyncTask(progressMsg, () => {
            const B = piece < modulo ? M + 1 : M;
            const typedB = new Float32Array(B * dim);
            for (let i = 0; i < B; ++i) {
                const vector = accessor(dataPoints[offset + i]);
                for (let d = 0; d < dim; ++d) {
                    typedB[i * dim + d] = vector[d];
                }
            }
            const partialMatrix = new weblas.pipeline.Tensor([B, dim], typedB);
            // Result is N x B matrix.
            const result = weblas.pipeline.sgemm(1, bigMatrix, partialMatrix, null, null);
            const partial = result.transfer();
            partialMatrix.delete();
            result.delete();
            progress += progressDiff;
            for (let i = 0; i < B; i++) {
                const kMin = new KMin<NearestEntry>(k);
                const iReal = offset + i;
                for (let j = 0; j < N; j++) {
                    if (j === iReal) {
                        continue;
                    }
                    const cosDist = 1 - partial[j * B + i];  // [j, i];
                    kMin.add(cosDist, {index: j, dist: cosDist});
                }
                nearest[iReal] = kMin.getMinKItems();
            }
            progress += progressDiff;
            offset += B;
            piece++;
        }, KNN_GPU_MSG_ID).then(() => {
            if (piece < numPieces) {
                step(resolve);
            } else {
                // logging.setModalMessage(null, KNN_GPU_MSG_ID);
                bigMatrix.delete();
                resolve(nearest);
            }
        }, () => {
            // GPU failed. Reverting back to CPU.
            // logging.setModalMessage(null, KNN_GPU_MSG_ID);
            const distFunc = (a: any, b: any, limit: any) => cosDistNorm(a, b);
            findKNN(dataPoints, k, accessor, distFunc).then((nearestVal: any) => {
                resolve(nearestVal);
            });
        });
    }
    return new Promise<NearestEntry[][]>((resolve) => step(resolve));
}

/**
 * Returns the K nearest neighbors for each vector where the distance
 * computation is done on the CPU using a user-specified distance method.
 *
 * @param dataPoints List of data points, where each data point holds an
 *   n-dimensional vector.
 * @param k Number of nearest neighbors to find.
 * @param accessor A method that returns the vector, given the data point.
 * @param dist Method that takes two vectors and a limit, and computes the
 *   distance between two vectors, with the ability to stop early if the
 *   distance is above the limit.
 */
export function findKNN<T>(dataPoints: T[], k: number, accessor: (dataPoint: T) => Float32Array, dist: (a: Vector, b: Vector, limit: number) => number): Promise<NearestEntry[][]> {
    return util.runAsyncTask<NearestEntry[][]>('Finding nearest neighbors...', () => {

        const N = dataPoints.length;
        const nearest: NearestEntry[][] = new Array(N);
        // Find the distances from node i.
        const kMin: Array<KMin<NearestEntry>> = new Array(N);

        for (let i = 0; i < N; i++) {
            kMin[i] = new KMin<NearestEntry>(k);
        }

        for (let i = 0; i < N; i++) {
            const a = accessor(dataPoints[i]);
            const kMinA = kMin[i];
            for (let j = i + 1; j < N; j++) {
                const kMinB = kMin[j];
                const limitI = kMinA.getSize() === k ?
                    kMinA.getLargestKey() || Number.MAX_VALUE :
                    Number.MAX_VALUE;
                const limitJ = kMinB.getSize() === k ?
                    kMinB.getLargestKey() || Number.MAX_VALUE :
                    Number.MAX_VALUE;
                const limit = Math.max(limitI, limitJ);
                const dist2ItoJ = dist(a, accessor(dataPoints[j]), limit);
                if (dist2ItoJ >= 0) {
                    kMinA.add(dist2ItoJ, {index: j, dist: dist2ItoJ});
                    kMinB.add(dist2ItoJ, {index: i, dist: dist2ItoJ});
                }
            }
        }

        for (let i = 0; i < N; i++) {
            nearest[i] = kMin[i].getMinKItems();
        }

        return nearest;
    });
}

/** Calculates the minimum distance between a search point and a rectangle. */
// function minDist(point: [number, number], x1: number, y1: number, x2: number, y2: number) {
//     const x = point[0];
//     const y = point[1];
//     const dx1 = x - x1;
//     const dx2 = x - x2;
//     const dy1 = y - y1;
//     const dy2 = y - y2;

//     if (dx1 * dx2 <= 0) {    // x is between x1 and x2
//         if (dy1 * dy2 <= 0) {  // (x,y) is inside the rectangle
//             return 0;            // return 0 as point is in rect
//         }
//         return Math.min(Math.abs(dy1), Math.abs(dy2));
//     }
//     if (dy1 * dy2 <= 0) {  // y is between y1 and y2
//         // We know it is already inside the rectangle
//         return Math.min(Math.abs(dx1), Math.abs(dx2));
//     }
//     let corner: [number, number];
//     if (x > x2) {
//         // Upper-right vs lower-right.
//         corner = y > y2 ? [x2, y2] : [x2, y1];
//     } else {
//         // Upper-left vs lower-left.
//         corner = y > y2 ? [x1, y2] : [x1, y1];
//     }

//     return Math.sqrt(dist22D([x, y], corner));
// }

/**
 * Returns the nearest neighbors of a particular point.
 *
 * @param dataPoints List of data points.
 * @param pointIndex The index of the point we need the nearest neighbors of.
 * @param k Number of nearest neighbors to search for.
 * @param accessor Method that maps a data point => vector (array of numbers).
 * @param distance Method that takes two vectors and returns their distance.
 */
export function findKNNofPoint<T>(
    dataPoints: T[], pointIndex: number, k: number,
    accessor: (dataPoint: T) => Float32Array,
    distance: (a: Vector, b: Vector) => number) {
    const kMin = new KMin<NearestEntry>(k);
    const a = accessor(dataPoints[pointIndex]);
    for (let i = 0; i < dataPoints.length; ++i) {
        if (i === pointIndex) {
            continue;
        }
        const b = accessor(dataPoints[i]);
        const dist = distance(a, b);
        kMin.add(dist, {index: i, dist});
    }
    return kMin.getMinKItems();
}