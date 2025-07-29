using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class AudioUtils
{
    public static float[] GetNormalizedAudioData(AudioClip clip)
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

        float maxVal = 0f;
        for (int i = 0; i < data.Length; i++)
        {
            if (Mathf.Abs(data[i]) > maxVal)
            {
                maxVal = Mathf.Abs(data[i]);
            }
        }
        if (maxVal > 0)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] /= maxVal;
            }
        }

        return data;
    }

    public static float[] Resample(float[] audioData, int fromSampleRate, int toSampleRate)
    {
        if (fromSampleRate == toSampleRate)
        {
            return audioData;
        }

        double ratio = (double)toSampleRate / fromSampleRate;
        int newLength = (int)(audioData.Length * ratio);
        float[] resampledData = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            float t = i / (float)ratio;
            float index = Mathf.Floor(t);
            float frac = t - index;

            int i0 = (int)index;
            int i1 = i0 + 1;

            if (i1 >= audioData.Length) i1 = audioData.Length - 1;

            resampledData[i] = Mathf.Lerp(audioData[i0], audioData[i1], frac);
        }

        return resampledData;
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
                float factor = (float)i / easeDurationFrames;
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