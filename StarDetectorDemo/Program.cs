using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp.Internal.Util;
using OpenCvSharp.Internal.Vectors;
using System.Buffers;

namespace StarDetectorDemo {
    class Program {
        static async Task Main(string[] args) {
            var path = @"C:\AP\LIGHT_2021-07-29_02-36-57_H_-9.90_300.00s_0151.tif";
            bool hotpixelFiltering = false;
            int noiseReductionRadius = 1;
            using (var t = new ResourcesTracker()) {
                var src = t.T(new Mat(path, ImreadModes.Unchanged));
                var statistics = ImageUtility.CalculateStatistics(src);
                var lut = t.T(ImageUtility.CreateMTFLookup(statistics));

                var starDetectionSrc = t.NewMat();
                ConvertToFloat(src, starDetectionSrc);
                if (hotpixelFiltering || noiseReductionRadius > 0) {
                    // Filter hotpixels before noise reduction, otherwise hotpixels may be promoted to stars
                    HotpixelFilter(starDetectionSrc, starDetectionSrc);
                }
                if (noiseReductionRadius > 0) {
                    var gaussianKernel = Cv2.GetGaussianKernel(1 + (1 << noiseReductionRadius), sigma: 2.0d);
                    Cv2.SepFilter2D(starDetectionSrc, starDetectionSrc, starDetectionSrc.Type(), gaussianKernel, gaussianKernel);
                }

                Mat structureMapSrc = starDetectionSrc;
                if (!hotpixelFiltering && noiseReductionRadius == 0) {
                    structureMapSrc = t.NewMat();
                    HotpixelFilter(starDetectionSrc, structureMapSrc);
                }

                var structureMap = t.NewMat();
                ComputeStructureMap(structureMapSrc, structureMap, structureLayers: 5);

                starDetectionSrc.SaveImage(@"C:\AP\star-detection-src.tif");
                structureMap.SaveImage(@"C:\AP\structure-map.tif");
                /*
                processed.SaveImage(@"C:\AP\processed.tif");
                processed.ConvertTo(processed, MatType.CV_16U);
                var processedStatistics = ImageUtility.CalculateStatistics(processed);
                var processedLut = t.T(ImageUtility.CreateMTFLookup(processedStatistics));

                var stretchedSrc = t.NewMat();
                ImageUtility.ApplyLUT(src, lut, stretchedSrc);
                ImageUtility.ApplyLUT(processed, processedLut, processed);

                var srcWindow = t.T(new Window("Original Image", stretchedSrc, WindowFlags.Normal | WindowFlags.KeepRatio));
                var processedWindow = t.T(new Window("Processed Image", processed, WindowFlags.Normal | WindowFlags.KeepRatio));
                Cv2.WaitKey();
                var srcVisible = Cv2.GetWindowProperty(srcWindow.Name, WindowPropertyFlags.Visible);
                if (srcVisible == 0) {
                    srcWindow.IsEnabledDispose = false;
                }

                var provessedVisible = Cv2.GetWindowProperty(processedWindow.Name, WindowPropertyFlags.Visible);
                if (provessedVisible == 0) {
                    processedWindow.IsEnabledDispose = false;
                }
                */

                Console.WriteLine();
            }
        }

        static void HotpixelFilter(Mat src, Mat dst) {
            // Hotpixel filtering with radius=1 uses a median morphological filter with a 3x3 box
            // If we want to support larger radii we'd have to implement a custom FilterEngine that takes a circular binary mask
            Cv2.MedianBlur(src, dst, 3);
        }

        static void ComputeStructureMap(Mat src, Mat dst, int structureLayers = 5) {
            if (src.Size() != dst.Size() || src.Type() != dst.Type()) {
                dst.Create(src.Size(), src.Type());
            }

            using (var tempImage = new Mat())
            using (var boxFilter = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3))) {
                // Step 1: Flatten the image with a high-pass filter
                var gaussianKernel = Cv2.GetGaussianKernel(1 + (1 << structureLayers), sigma: 5.3d);

                src.SaveImage(@"C:\AP\structure-src.tif");

                Cv2.SepFilter2D(src, tempImage, src.Type(), gaussianKernel, gaussianKernel);
                tempImage.SaveImage(@"C:\AP\structure-blurred.tif");
                using (var t = src.Subtract(tempImage).ToMat()) {
                    ImageUtility.Clip(t);
                    unsafe {
                        var imageData = (float*)t.DataPointer;
                        var numPixels = t.Rows * t.Cols;
                        float min = float.MaxValue;
                        float max = -1.0f;
                        for (int i = 0; i < numPixels; ++i) {
                            var pixel = imageData[i];
                            min = pixel < min ? pixel : min;
                            max = pixel > max ? pixel : max;
                        }
                        Console.WriteLine($"MAX: {max}, MIN: {min}");
                        using (var u = t.Subtract(min).Divide(max - min).ToMat()) {
                            u.CopyTo(tempImage);
                        }
                    }
                }

                tempImage.SaveImage(@"C:\AP\structure-step-1.tif");

                // Step 2: Boost small structures with a dilation box filter
                Cv2.MorphologyEx(tempImage, tempImage, MorphTypes.Dilate, boxFilter);

                tempImage.SaveImage(@"C:\AP\structure-step-2.tif");

                // Step 3: Binarize foreground structures based on noise estimates
                // TODO: Dispose wavelet layers afterwards
                var noiseClippingMultiplier = 3.0d;

                // The 2nd wavelet layer is a cleaner image more likely to contain stars and less likely to include noise
                var waveletLayers = ImageUtility.ComputeAtrousB3SplineDyadicWaveletLayers(tempImage, 2);
                long numBackgroundPixels;
                var ksigmaNoise = ImageUtility.KappaSigmaNoiseEstimate(waveletLayers[1], out numBackgroundPixels, clippingMultipler: noiseClippingMultiplier) / 0.2007f;
                var backgroundPercentage = (double)numBackgroundPixels / (tempImage.Rows * tempImage.Cols);
                Console.WriteLine($"KSigma Noise Estimate: {ksigmaNoise}. Background Pixels = {backgroundPercentage}");

                // TODO: This median calculation is wrong, since tempImage is float
                double dilatedMedian;
                using (var medianMat = new Mat()) {
                    float[] data;
                    tempImage.GetArray<float>(out data);

                    Array.Sort(data);
                    // ConvertToUShort(tempImage, medianMat);
                    // var med = ImageUtility.CalculateStatistics(medianMat).Median;
                    var med = data[data.Length >> 1];
                    Console.WriteLine($"Median is {med}");
                    // dilatedMedian = med / ushort.MaxValue;
                    dilatedMedian = med;
                }
                Console.WriteLine($"Median: {dilatedMedian}, Threshold = {dilatedMedian + noiseClippingMultiplier * ksigmaNoise}");
                Binarize(tempImage, tempImage, (float)(dilatedMedian + noiseClippingMultiplier * ksigmaNoise));

                tempImage.SaveImage(@"C:\AP\structure-step-3.tif");

                tempImage.CopyTo(dst);
            }
        }

        static void ConvertToFloat(Mat src, Mat dst) {
            if (src.Size() != dst.Size() || dst.Type() != MatType.CV_32F) {
                dst.Create(src.Size(), MatType.CV_32F);
            }
            unsafe {
                var srcData = (ushort*)src.DataPointer;
                var dstData = (float*)dst.DataPointer;
                var numPixels = src.Rows * src.Cols;
                var maxShort = (float)ushort.MaxValue;
                for (int i = 0; i < numPixels; ++i) {
                    dstData[i] = (float)srcData[i] / maxShort;
                }
            }
        }

        static void Binarize(Mat src, Mat dst, float threshold) {
            if (src.Size() != dst.Size() || src.Type() != dst.Type()) {
                dst.Create(src.Size(), src.Type());
            }
            unsafe {
                var srcData = (float*)src.DataPointer;
                var dstData = (float*)dst.DataPointer;
                var numPixels = src.Rows * src.Cols;
                for (int i = 0; i < numPixels; ++i) {
                    dstData[i] = srcData[i] >= threshold ? 1.0f : 0.0f;
                }
            }
        }

        static void ConvertToUShort(Mat src, Mat dst) {
            // TODO: Throw if src == dst reference
            if (src.Size() != dst.Size() || dst.Type() != MatType.CV_16U) {
                dst.Create(src.Size(), MatType.CV_16U);
            }
            unsafe {
                var srcData = (float*)src.DataPointer;
                var dstData = (ushort*)dst.DataPointer;
                var numPixels = src.Rows * src.Cols;
                for (int i = 0; i < numPixels; ++i) {
                    dstData[i] = (ushort)(Math.Ceiling(srcData[i] * ushort.MaxValue));
                }
            }
        }

        static void GetBicubicWaveletFilter(Mat src, Mat dst, int layer) {
            var bicubicWavelet2D = new Mat(new Size(1, 5), MatType.CV_32F);
            unsafe {
                var data = (float*)bicubicWavelet2D.DataPointer;
                data[0] = data[4] = 0.0625f;
                data[1] = data[3] = 0.25f;
                data[2] = 0.375f;
            }

            src.SaveImage(@"C:\AP\src.tif");

            var srcToFloat = new Mat();
            ConvertToFloat(src, srcToFloat);

            Cv2.ImWrite(@"C:\AP\src_to_float.tif", srcToFloat);
            //csrcToFloat.SaveImage();

            var convolved = new Mat();
            Cv2.SepFilter2D(srcToFloat, convolved, MatType.CV_32F, bicubicWavelet2D, bicubicWavelet2D, borderType: BorderTypes.Reflect);

            convolved.SaveImage(@"C:\AP\convolved.tif");

            var result = srcToFloat.Subtract(convolved).ToMat();
            /*
            var result = new Mat(srcToFloat.Size(), MatType.CV_32F);
            unsafe {
                var srcData = (float*)srcToFloat.DataPointer;
                var convolvedData = (float*)convolved.DataPointer;
                var resultData = (float*)result.DataPointer;
                var numPixels = result.Rows * result.Cols;
                for (int i = 0; i < numPixels; ++i) {
                    if (srcData[i] <= convolvedData[i]) {
                        resultData[i] = srcData[i];
                    } else {
                        resultData[i] = 0.0f;
                    }
                }
            }
            */

            unsafe {
                var resultData = (float*)result.DataPointer;
                var numPixels = result.Rows * result.Cols;
                for (int i = 0; i < numPixels; ++i) {
                    resultData[i] = Math.Min(1.0f, Math.Max(0.0f, resultData[i]));
                }
            }

            result.SaveImage(@"C:\AP\subtracted.tif");
            result.CopyTo(dst); // .ConvertTo(dst, MatType.CV_16U);
        }
    }
}
