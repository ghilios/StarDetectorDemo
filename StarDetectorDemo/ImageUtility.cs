using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StarDetectorDemo {
    public static class ImageUtility {
        public class Statistics {
            public double Median { get; set; }
            public double MAD { get; set; }
        }

        private static double NormalizeUShort(double val, int bitDepth) {
            return val / (double)(1 << bitDepth);
        }

        private static ushort DenormalizeUShort(double val) {
            return (ushort)(val * ushort.MaxValue);
        }

        private static double MidtonesTransferFunction(double midToneBalance, double x) {
            if (x > 0) {
                if (x < 1) {
                    return (midToneBalance - 1) * x / ((2 * midToneBalance - 1) * x - midToneBalance);
                }
                return 1;
            }
            return 0;
        }

        public static void ApplyLUT(Mat src, Mat lut, Mat dst) {
            if (dst.Size() != src.Size() || dst.Type() != src.Type()) {
                dst.Create(src.Size(), src.Type());
            }

            unsafe {
                var srcData = (ushort*)src.DataPointer;
                var dstData = (ushort*)dst.DataPointer;
                var lutData = (ushort*)lut.DataPointer;
                var numPixels = src.Rows * src.Cols;
                for (int i = 0; i < numPixels; ++i) {
                    dstData[i] = lutData[srcData[i]];
                }
            }
        }

        public static Mat CreateMTFLookup(Statistics statistics, double shadowsClipping = -2.8, double targetHistogramMedianPercent = 0.2) {
            var normalizedMedian = NormalizeUShort(statistics.Median, 16);
            var normalizedMAD = NormalizeUShort(statistics.MAD, 16);

            var scaleFactor = 1.4826; // see https://en.wikipedia.org/wiki/Median_absolute_deviation

            double shadows = normalizedMedian + shadowsClipping * normalizedMAD * scaleFactor;
            double midtones = MidtonesTransferFunction(targetHistogramMedianPercent, normalizedMedian - shadows);
            double highlights = 1;

            var lut = new Mat(1, ushort.MaxValue, MatType.CV_16U);
            unsafe {
                var lutData = (ushort*)lut.DataPointer;
                for (int i = 0; i < ushort.MaxValue; i++) {
                    double value = NormalizeUShort(i, 16);
                    lutData[i] = DenormalizeUShort(MidtonesTransferFunction(midtones, 1 - highlights + value - shadows));
                }
            }

            return lut;
        }

        private static uint[] CalculateHistogram(Mat image) {
            var histogram = new uint[ushort.MaxValue + 1];
            unsafe {
                var dataPtr = (ushort*)image.DataPointer;
                for (var i = 0; i < image.Rows * image.Cols; ++i) {
                    histogram[dataPtr[i]] += 1;
                }
            }
            return histogram;
        }

        public static Statistics CalculateStatistics(Mat image) {
            var histogram = CalculateHistogram(image);
            var targetMedianCount = image.Rows * image.Cols / 2.0;
            uint currentCount = 0;
            double median = -1.0d;
            for (uint i = 0; i <= ushort.MaxValue; ++i) {
                currentCount += histogram[i];
                if (currentCount > targetMedianCount) {
                    median = i;
                    break;
                } else if (currentCount == targetMedianCount) {
                    for (uint j = i + 1; i <= ushort.MaxValue; ++i) {
                        if (histogram[j] > 0) {
                            median = (i + j) / 2.0d;
                            break;
                        }
                    }
                    break;
                }
            }

            currentCount = 0;
            int upIndex = (int)Math.Ceiling(median);
            int downIndex = (int)Math.Floor(median);
            double beforeMedian = -1;
            double mad;
            while (true) {
                while (upIndex <= ushort.MaxValue && histogram[upIndex] == 0) {
                    ++upIndex;
                }
                while (downIndex >= 0 && histogram[downIndex] == 0) {
                    --downIndex;
                }

                var upDistance = upIndex <= ushort.MaxValue ? Math.Abs(upIndex - median) : double.MaxValue;
                var downDistance = downIndex >= 0 ? Math.Abs(downIndex - median) : double.MaxValue;
                int chosenIndex;
                if (upDistance <= downDistance) {
                    chosenIndex = upIndex;
                    currentCount += histogram[upIndex++];
                } else {
                    chosenIndex = downIndex;
                    currentCount += histogram[downIndex--];
                }

                if (currentCount == targetMedianCount) {
                    beforeMedian = Math.Abs(chosenIndex - median);
                } else if (currentCount > targetMedianCount) {
                    if (beforeMedian >= 0) {
                        mad = (beforeMedian + Math.Abs(chosenIndex - median)) / 2.0d;
                    } else {
                        mad = Math.Abs(chosenIndex - median);
                    }
                    break;
                }
            }

            return new Statistics() { Median = median, MAD = mad };
        }

        private static Mat SeparatedB3SplineScalingFilter(int dyadicLayer) {
            /*
             * Cubic Spline coefficients
             *   1.0/256, 1.0/64, 3.0/128, 1.0/64, 1.0/256,
             *   1.0/64,  1.0/16, 3.0/32,  1.0/16, 1.0/64,
             *   3.0/128, 3.0/32, 9.0/64,  3.0/32, 3.0/128,
             *   1.0/64,  1.0/16, 3.0/32,  1.0/16, 1.0/64,
             *   1.0/256, 1.0/64, 3.0/128, 1.0/64, 1.0/256
             *   
             * Decomposed 1D filter
             *   0.0625,  0.25,   0.375,   0.25,   0.0625
             */

            int size = (1 << (dyadicLayer + 2)) + 1;
            var bicubicWavelet2D = new Mat(new Size(1, size), MatType.CV_32F, 0.0d);
            // Each successive layer is downsampled 2x. Rather than copy the matrix to convolve it, we can padd
            // the separated filter with zeroes
            unsafe {
                var data = (float*)bicubicWavelet2D.DataPointer;
                data[0] = data[size - 1] = 0.0625f;
                data[1 << dyadicLayer] = data[size - (1 << dyadicLayer) - 1] = 0.25f;
                data[size >> 1] = 0.375f;
            }
            return bicubicWavelet2D;
        }

        public static Mat Clip(Mat src, float min = 0.0f, float max = 1.0f) {
            unsafe {
                var srcData = (float*)src.DataPointer;
                var numPixels = src.Rows * src.Cols;
                for (int i = 0; i < numPixels; ++i) {
                    srcData[i] = Math.Min(1.0f, Math.Max(0.0f, srcData[i]));
                }
                return src;
            }
        }

        public static Mat[] ComputeAtrousB3SplineDyadicWaveletLayers(Mat src, int numLayers) {
            var layers = new Mat[numLayers + 1];
            var previousLayer = src;
            Mat convolved = new Mat();
            for (int i = 0; i < numLayers; ++i) {
                var scalingFilter = SeparatedB3SplineScalingFilter(i);
                Cv2.SepFilter2D(previousLayer, convolved, MatType.CV_32F, scalingFilter, scalingFilter, borderType: BorderTypes.Reflect);
                var nextLayer = previousLayer.Subtract(convolved).ToMat();
                previousLayer = layers[i] = Clip(nextLayer);
            }
            layers[numLayers] = convolved;
            return layers;
        }

        public static double KappaSigmaNoiseEstimate(Mat src, out long numBackgroundPixels, double clippingMultipler = 3.0d, double allowedError = 0.05, int maxIterations = 10) {
            // NOTE: This algorithm could be sped up for 16-byte integer data by building a histogram. Consider this if performance becomes problematic
            // TODO: Assert input data are float

            // Gaussian noise scaling factors:
            //  { 0.8907F, 0.2007F, 0.0856F, 0.0413F, 0.0205F, 0.0103F, 0.0052F, 0.0026F, 0.0013F, 0.0007F };
            unsafe {
                var srcData = (float*)src.DataPointer;
                int numPixels = src.Rows * src.Cols;
                var threshold = float.MaxValue;
                var lastSigma = 1.0d;
                var backgroundPixels = 0L;
                int numIterations = 0;
                while (numIterations < maxIterations) {
                    ++numIterations;
                    var total = 0.0d;
                    backgroundPixels = 0L;
                    for (int i = 0; i < numPixels; ++i) {
                        var pixel = srcData[i];
                        if (pixel < threshold && pixel > 0.0f) {
                            total += pixel;
                            ++backgroundPixels;
                        }
                    }

                    var mean = total / backgroundPixels;
                    var mrs = 0.0d;
                    for (int i = 0; i < numPixels; ++i) {
                        var pixel = srcData[i];
                        if (pixel < threshold) {
                            var error = pixel - mean;
                            mrs += error * error;
                        }
                    }

                    var variance = mrs / (backgroundPixels - 1);
                    var sigma = Math.Sqrt(variance);
                    if (numIterations > 1) {
                        var sigmaConvergenceError = Math.Abs(sigma - lastSigma) / lastSigma;
                        if (sigmaConvergenceError <= allowedError) {
                            lastSigma = sigma;
                            break;
                        }
                    }
                    threshold = (float)(mean + clippingMultipler * sigma);
                    lastSigma = sigma;
                }

                numBackgroundPixels = backgroundPixels;
                return lastSigma;
            }
        }
    }
}
