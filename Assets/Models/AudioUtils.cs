using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class AudioUtils
{
    /// <summary>Feature extractor needs at least 9 STFT frames (same as extract_features.py).</summary>
    public const int MinSamples88k = 8 * 735;

    /// <summary>
    /// Mirrors the Python reference loader (librosa.load(sr=88200) + peak normalisation):
    /// downmix to mono, resample with a band-limited sinc filter, THEN normalise to peak 1.
    /// </summary>
    public static float[] LoadForFeatureExtraction(AudioClip clip, int targetSampleRate = 88200)
    {
        float[] mono = GetMonoAudioData(clip);
        if (mono == null) return null;
        float[] resampled = Resample(mono, clip.frequency, targetSampleRate);
        NormalizePeak(resampled);
        return resampled;
    }

    public static float[] GetMonoAudioData(AudioClip clip)
    {
        if (clip == null) return null;

        float[] data = new float[clip.samples * clip.channels];
        clip.GetData(data, 0);

        if (clip.channels > 1)
        {
            float[] monoData = new float[clip.samples];
            for (int i = 0; i < clip.samples; i++)
            {
                float currentSample = 0;
                for (int c = 0; c < clip.channels; c++)
                {
                    currentSample += data[i * clip.channels + c];
                }
                monoData[i] = currentSample / clip.channels;
            }
            data = monoData;
        }
        return data;
    }

    public static void NormalizePeak(float[] data)
    {
        float maxVal = 0f;
        for (int i = 0; i < data.Length; i++)
            maxVal = Mathf.Max(maxVal, Mathf.Abs(data[i]));
        if (maxVal > 0)
        {
            float inv = 1f / maxVal;
            for (int i = 0; i < data.Length; i++) data[i] *= inv;
        }
    }

    /// <summary>Kept for backwards compatibility: mono + peak normalisation at the clip's own rate.</summary>
    public static float[] GetNormalizedAudioData(AudioClip clip)
    {
        float[] data = GetMonoAudioData(clip);
        if (data != null) NormalizePeak(data);
        return data;
    }

    // Band-limited rational resampler, equivalent to scipy.signal.resample_poly(x, up, down,
    // window=('kaiser', KaiserBeta)) with a longer filter (ZeroCrossings taps per side).
    // Linear interpolation leaves strong images above the source Nyquist that the 88.2 kHz
    // MFCC / autocorrelation features see; this filter tracks librosa's soxr_hq within ~1e-3
    // on the final blendshapes (linear interpolation: up to ~4.6).
    const int ZeroCrossings = 32;
    const double KaiserBeta = 12.0;

    public static float[] Resample(float[] audioData, int fromSampleRate, int toSampleRate)
    {
        if (fromSampleRate == toSampleRate || audioData == null || audioData.Length == 0)
            return audioData;

        int g = Gcd(fromSampleRate, toSampleRate);
        int up = toSampleRate / g;
        int down = fromSampleRate / g;
        int maxRate = Math.Max(up, down);
        int halfLen = ZeroCrossings * maxRate;
        int taps = 2 * halfLen + 1;

        // firwin(taps, cutoff=1/maxRate, window=('kaiser', beta)), DC gain normalised, times `up`.
        double cutoff = 1.0 / maxRate;
        double i0Beta = BesselI0(KaiserBeta);
        var h = new double[taps];
        double sum = 0;
        for (int n = 0; n < taps; n++)
        {
            double m = n - halfLen;
            double x = cutoff * m;
            double sinc = m == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
            double r = 2.0 * n / (taps - 1) - 1.0;
            double w = BesselI0(KaiserBeta * Math.Sqrt(Math.Max(0.0, 1.0 - r * r))) / i0Beta;
            h[n] = cutoff * sinc * w;
            sum += h[n];
        }
        float[] hf = new float[taps];
        for (int n = 0; n < taps; n++) hf[n] = (float)(h[n] / sum * up);

        int inLen = audioData.Length;
        int outLen = (int)(((long)inLen * up + down - 1) / down);
        float[] output = new float[outLen];
        for (int j = 0; j < outLen; j++)
        {
            // y[j] = sum_k x[k] * h[j*down + halfLen - k*up]
            long c = (long)j * down + halfLen;
            long kMin = Math.Max(0L, CeilDiv(c - (taps - 1), up));
            long kMax = Math.Min(inLen - 1L, c / up);
            double acc = 0;
            for (long k = kMin; k <= kMax; k++)
                acc += audioData[k] * hf[c - k * up];
            output[j] = (float)acc;
        }
        return output;
    }

    static long CeilDiv(long a, long b) => a >= 0 ? (a + b - 1) / b : -((-a) / b);

    static int Gcd(int a, int b)
    {
        while (b != 0) { int t = a % b; a = b; b = t; }
        return a;
    }

    static double BesselI0(double x)
    {
        double sum = 1.0, term = 1.0, q = x * x / 4.0;
        for (int k = 1; k < 500; k++)
        {
            term *= q / ((double)k * k);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }
}

public class AudioFeatureProcessor
{
    private readonly int _frameSize;
    private readonly int _overlap;

    public AudioFeatureProcessor(int frameSize = 128, int overlap = 32)
    {
        _frameSize = frameSize;
        _overlap = overlap;
    }

    public float[,] ProcessFeatures(
        float[] featureArray, 
        int numFrames, 
        int numFeatures,
        Func<float[,], float[,]> runInference)
    {
        var allDecodedOutputs = new List<float[,]>();
        
        int startIdx = 0;
        while (startIdx < numFrames)
        {
            int endIdx = Mathf.Min(startIdx + _frameSize, numFrames);
            int chunkLen = endIdx - startIdx;

            float[,] chunk = GetChunk(featureArray, startIdx, chunkLen, numFeatures);
            float[,] paddedChunk = PadChunk(chunk, numFeatures);

            float[,] decodedOutputs = runInference(paddedChunk);

            float[,] trimmedDecodedOutputs = new float[chunkLen, decodedOutputs.GetLength(1)];
            Array.Copy(decodedOutputs, trimmedDecodedOutputs, chunkLen * decodedOutputs.GetLength(1));
            
            if (allDecodedOutputs.Count > 0)
            {
                float[,] lastChunk = allDecodedOutputs.Last();
                allDecodedOutputs.RemoveAt(allDecodedOutputs.Count - 1);
                float[,] blendedChunk = BlendChunks(lastChunk, trimmedDecodedOutputs);
                allDecodedOutputs.Add(blendedChunk);
            }
            else
            {
                allDecodedOutputs.Add(trimmedDecodedOutputs);
            }

            startIdx += _frameSize - _overlap;
        }

        float[,] finalData = ConcatenateChunks(allDecodedOutputs, numFrames);
        PostProcess(finalData);

        return finalData;
    }
    
    private float[,] GetChunk(float[] featureArray, int start, int length, int numFeatures)
    {
        float[,] chunk = new float[length, numFeatures];

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                int sourceIndex = (start + i) * numFeatures + j;
                
                if (sourceIndex < featureArray.Length)
                {
                    chunk[i, j] = featureArray[sourceIndex];
                }
            }
        }
        return chunk;
    }
    
    private float[,] PadChunk(float[,] chunk, int numFeatures)
    {
        int currentLength = chunk.GetLength(0);
        if (currentLength >= _frameSize) return chunk;

        float[,] paddedChunk = new float[_frameSize, numFeatures];
        Array.Copy(chunk, paddedChunk, chunk.Length);

        int padLength = _frameSize - currentLength;
        for (int i = 0; i < padLength; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                paddedChunk[currentLength + i, j] = chunk[currentLength - 1, j];
            }
        }
        return paddedChunk;
    }

    private float[,] BlendChunks(float[,] chunk1, float[,] chunk2)
    {
        int len1 = chunk1.GetLength(0);
        int len2 = chunk2.GetLength(0);
        int numFeatures = chunk1.GetLength(1);
        
        int actualOverlap = Mathf.Min(_overlap, len1, len2);
        if (actualOverlap <= 0) return Concatenate(chunk1, chunk2);
        
        float[,] blendedChunk1 = (float[,])chunk1.Clone();

        for (int i = 0; i < actualOverlap; i++)
        {
            float alpha = (float)i / actualOverlap;
            for(int j = 0; j < numFeatures; j++)
            {
                int blendIndex = len1 - actualOverlap + i;
                blendedChunk1[blendIndex, j] = Mathf.Lerp(chunk1[blendIndex, j], chunk2[i, j], alpha);
            }
        }
        
        int remainingLength = len2 - actualOverlap;
        float[,] remainingChunk2 = new float[remainingLength, numFeatures];
        Array.Copy(chunk2, actualOverlap * numFeatures, remainingChunk2, 0, remainingLength * numFeatures);

        return Concatenate(blendedChunk1, remainingChunk2);
    }
    
    private void PostProcess(float[,] data)
    {
        int numFrames = data.GetLength(0);
        int numFeatures = data.GetLength(1);

        for (int i = 0; i < numFrames; i++)
            for (int j = 0; j < Mathf.Min(61, numFeatures); j++)
                data[i, j] /= 100.0f;

        int easeDurationFrames = Mathf.Min((int)(0.1f * 60), numFrames);
        if (easeDurationFrames > 0)
        {
            for (int i = 0; i < easeDurationFrames; i++)
            {
                // np.linspace(0, 1, n): last eased frame reaches exactly 1
                float factor = easeDurationFrames > 1 ? (float)i / (easeDurationFrames - 1) : 0f;
                for (int j = 0; j < numFeatures; j++) data[i, j] *= factor;
            }
        }
        
        int[] columnsToZero = { 0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 };
        for (int i = 0; i < numFrames; i++)
            foreach (int col in columnsToZero)
                if (col < numFeatures) data[i, col] = 0;
    }

    private float[,] ConcatenateChunks(List<float[,]> chunks, int totalFrames)
    {
        if (chunks.Count == 0) return new float[0, 0];
        int numFeatures = chunks[0].GetLength(1);
        float[,] result = new float[totalFrames, numFeatures];
        int currentRow = 0;
        foreach (var chunk in chunks)
        {
            int chunkRows = chunk.GetLength(0);
            if (currentRow + chunkRows > totalFrames) chunkRows = totalFrames - currentRow;
            if (chunkRows <= 0) break;
            
            Array.Copy(chunk, 0, result, currentRow * numFeatures, chunkRows * numFeatures);
            currentRow += chunkRows;
        }
        return result;
    }

    private float[,] Concatenate(float[,] arr1, float[,] arr2)
    {
        int rows1 = arr1.GetLength(0), cols = arr1.GetLength(1);
        int rows2 = arr2.GetLength(0);
        float[,] result = new float[rows1 + rows2, cols];
        Array.Copy(arr1, 0, result, 0, arr1.Length);
        Array.Copy(arr2, 0, result, arr1.Length, arr2.Length);
        return result;
    }
}