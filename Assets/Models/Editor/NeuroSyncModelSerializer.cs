using System.IO;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

// Tools > NeuroSync > Serialize Selected ONNX to StreamingAssets
// Writes <name>.sentis (fp32) and <name>_fp16.sentis (Float16 weights) for each selected ONNX model.
public static class NeuroSyncModelSerializer
{
    [MenuItem("Tools/NeuroSync/Serialize Selected ONNX to StreamingAssets")]
    static void SerializeSelected()
    {
        Directory.CreateDirectory(Application.streamingAssetsPath);
        foreach (var obj in Selection.objects)
        {
            if (obj is not ModelAsset asset) continue;
            string baseName = asset.name;

            var model = ModelLoader.Load(asset);
            string fp32Path = Path.Combine(Application.streamingAssetsPath, baseName + ".sentis");
            ModelWriter.Save(fp32Path, model);

            var quantized = ModelLoader.Load(asset);
            ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref quantized);
            string fp16Path = Path.Combine(Application.streamingAssetsPath, baseName + "_fp16.sentis");
            ModelWriter.Save(fp16Path, quantized);

            Debug.Log($"[NeuroSync] Saved {fp32Path} and {fp16Path}");
        }
        AssetDatabase.Refresh();
    }

    [MenuItem("Tools/NeuroSync/Serialize Selected ONNX to StreamingAssets", true)]
    static bool Validate() => Selection.activeObject is ModelAsset;
}
