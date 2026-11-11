using UnityEngine;
using Unity.InferenceEngine;
using System;
using System.Collections.Generic;
using System.IO;

public class NeurosuncController : MonoBehaviour
{
    public enum ProcessingMode
    {
        OnDevice,
        API
    }

    public ProcessingMode processingMode = ProcessingMode.OnDevice;
    private int targetSampleRate = 88200;

    public AudioClip inputAudioClip;

    public BlendshapeManager blendshapeManager;

    [Header("On-device models (StreamingAssets)")]
    // Keep the feature extractor fp32: Float16 weights on its DFT/mel matrices roughly double the
    // blendshape error vs. the Python reference (~1e-2 instead of ~1e-5). Decoder fp16 is fine.
    public string featureExtractorFileName = "audio_feature_extractor.sentis";
    public string neurosyncFileName = "neurosync_fp16.sentis";
    public BackendType featureBackend = BackendType.GPUCompute;
    public BackendType decoderBackend = BackendType.GPUCompute;

    private string featureExtractorModelPath;
    private string neurosyncModelPath;
    private Worker _featureWorker;
    private Worker _decoderWorker;
    private AudioFeatureProcessor _featureProcessor;
    
    private List<float[]> _generatedBlendshapes;
    private AudioClip _generatedAudioClip;
    private bool _isDataReady = false;

    void Start()
    {
        if (blendshapeManager == null)
        {
            Debug.LogError("BlendshapeManager is not assigned!", this);
            return;
        }

        if (processingMode == ProcessingMode.OnDevice)
        {
            InitializeOnDeviceModels();
        }

        if (inputAudioClip != null)
        {
            ProcessAudioClip();
        }
    }

    private void InitializeOnDeviceModels()
    {
        featureExtractorModelPath = Path.Combine(Application.streamingAssetsPath, featureExtractorFileName);
        neurosyncModelPath = Path.Combine(Application.streamingAssetsPath, neurosyncFileName);
        
        var featureModel = ModelLoader.Load(featureExtractorModelPath);
        _featureWorker = new Worker(featureModel, ResolveBackend(featureBackend));
        var decoderModel = ModelLoader.Load(neurosyncModelPath);
        _decoderWorker = new Worker(decoderModel, ResolveBackend(decoderBackend));
        _featureProcessor = new AudioFeatureProcessor();
        Debug.Log("On-device models initialized successfully.");
    }

    private static BackendType ResolveBackend(BackendType requested)
    {
        return requested == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders ? BackendType.CPU : requested;
    }

    private void ProcessAudioClip()
    {
        switch (processingMode)
        {
            case ProcessingMode.OnDevice:
                GenerateAndStoreData(inputAudioClip);
                break;
            case ProcessingMode.API:
                _ = blendshapeManager.WavToBlendshape(inputAudioClip);
                break;
        }
    }

    void Update()
    {
        if (_isDataReady && Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Spacebar pressed! Starting blendshape playback.");
            blendshapeManager.SetBlendshapeDataAndPlay(_generatedBlendshapes, _generatedAudioClip);
            _isDataReady = false; 
        }
    }
    
    public void GenerateAndStoreData(AudioClip audioClip)
    {
        if (audioClip == null)
        {
            Debug.LogError("Input audio clip is missing!", this);
            return;
        }

        Debug.Log($"Starting on-device blendshape generation for audio: '{audioClip.name}'.");
        
        // Same order as the Python reference: mono -> band-limited resample to 88.2 kHz -> peak normalise.
        float[] resampledAudio = AudioUtils.LoadForFeatureExtraction(audioClip, targetSampleRate);
        if (resampledAudio.Length < AudioUtils.MinSamples88k)
        {
            Debug.LogError($"Audio clip is too short for feature extraction ({resampledAudio.Length} samples at {targetSampleRate} Hz, need {AudioUtils.MinSamples88k}).", this);
            return;
        }
        
        using var audioTensor = new Tensor<float>(new TensorShape(1, resampledAudio.Length), resampledAudio);
        _featureWorker.Schedule(audioTensor);
        using var featureTensor = (_featureWorker.PeekOutput() as Tensor<float>).ReadbackAndClone();
        float[] featureData = featureTensor.DownloadToArray();
        int numFrames = featureTensor.shape[0];
        int numFeatures = featureTensor.shape[1];
        
        Func<float[,], float[,]> runDecoderInference = (paddedChunk) =>
        {
            int frameSize = paddedChunk.GetLength(0);
            int features = paddedChunk.GetLength(1);
            float[] flatPaddedChunk = new float[frameSize * features];
            Buffer.BlockCopy(paddedChunk, 0, flatPaddedChunk, 0, flatPaddedChunk.Length * sizeof(float));
            
            using var inputTensor = new Tensor<float>(new TensorShape(1, frameSize, features), flatPaddedChunk);
            _decoderWorker.Schedule(inputTensor);
            using var outputTensor = (_decoderWorker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            float[] flatOutput = outputTensor.DownloadToArray();
            var outputArray = new float[frameSize, outputTensor.shape[2]];

            for (int i = 0; i < frameSize; i++)
                for (int j = 0; j < outputTensor.shape[2]; j++)
                    outputArray[i, j] = flatOutput[i * outputTensor.shape[2] + j];
            
            return outputArray;
        };
        
        float[,] generatedData = _featureProcessor.ProcessFeatures(
            featureData, numFrames, numFeatures, runDecoderInference);
        
        if (generatedData != null)
        {
            _generatedBlendshapes = ConvertToList(generatedData);
            _generatedAudioClip = audioClip;
            _isDataReady = true;
            Debug.Log($"On-device blendshape data generation complete! Press Spacebar to play.");
        }
    }
    
    private List<float[]> ConvertToList(float[,] array2D)
    {
        var list = new List<float[]>();
        int rows = array2D.GetLength(0);
        int cols = array2D.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            var row = new float[cols];
            Buffer.BlockCopy(array2D, i * cols * sizeof(float), row, 0, cols * sizeof(float));
            list.Add(row);
        }
        return list;
    }

    void OnDestroy()
    {
        _featureWorker?.Dispose();
        _decoderWorker?.Dispose();
    }
}