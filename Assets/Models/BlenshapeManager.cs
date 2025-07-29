using UnityEngine;
using UnityEngine.Networking;
using System.Threading.Tasks;
using System.Text;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

public class BlendshapeManager : MonoBehaviour
{
    [SerializeField] private string audioToBlendshapeUrl = "http://localhost:5005/audio_to_blendshapes";
    [SerializeField] private SkinnedMeshRenderer m_SkinnedMesh;
    [SerializeField] private float m_BlendShapeScale = 110f;

    private const int TARGET_FRAMERATE = 60;
    private const int ARKIT_BLENDSHAPE_COUNT = 52;

    private bool isPlaying = false;
    private List<BlendshapeFrame> blendshapeFrames = new List<BlendshapeFrame>();
    private AudioSource audioSource;
    private float playbackTime = 0f;

    void Awake()
    {
        Application.targetFrameRate = TARGET_FRAMERATE;
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
        }
    }

    public void SetBlendshapeDataAndPlay(List<float[]> blendshapes, AudioClip audioClip)
    {
        if (blendshapes == null || blendshapes.Count == 0)
        {
            Debug.LogError("No blendshape data received.", this);
            return;
        }
        
        blendshapeFrames = ConvertToBlendshapeFrames(blendshapes);
        audioSource.clip = audioClip;
        Play();
    }
    
    void Update()
    {
        if (!isPlaying || blendshapeFrames.Count == 0)
        {
            return;
        }

        playbackTime += Time.deltaTime;

        int targetFrame = Mathf.FloorToInt(playbackTime * TARGET_FRAMERATE);

        if (targetFrame >= blendshapeFrames.Count)
        {
            Stop();
            return;
        }

        ApplyBlendshapes(blendshapeFrames[targetFrame]);
    }

    private void ApplyBlendshapes(BlendshapeFrame frame)
    {
        if (m_SkinnedMesh == null) return;

        foreach (var kvp in frame.values)
        {
            string blendshapeName = kvp.Key;
            float value = kvp.Value;

            int blendshapeIndex = m_SkinnedMesh.sharedMesh.GetBlendShapeIndex(blendshapeName);

            if (blendshapeIndex != -1)
            {
                //Note: Due to the difference in eyeWide_Left, we set it to follow the eyeWide_Right value.
                if (blendshapeName.Contains("eyeWide_Left"))
                {
                    int blendshapeIndexTemp = m_SkinnedMesh.sharedMesh.GetBlendShapeIndex("eyeWide_Right");
                    frame.values.TryGetValue("eyeWide_Right", out float eyeWideRightValue);
                    value = eyeWideRightValue;
                    m_SkinnedMesh.SetBlendShapeWeight(blendshapeIndexTemp, value * m_BlendShapeScale);
                }
                
                m_SkinnedMesh.SetBlendShapeWeight(blendshapeIndex, value * m_BlendShapeScale);
            }
        }
    }

    private void Play()
    {
        if (blendshapeFrames.Count == 0)
        {
            Debug.LogWarning("No blendshape data loaded!");
            return;
        }

        if (audioSource != null && audioSource.clip != null)
        {
            playbackTime = 0f;
            audioSource.Play();
            isPlaying = true;
        }
    }

    public void Stop()
    {
        isPlaying = false;
        playbackTime = 0f;
        
        if (m_SkinnedMesh != null)
        {
            for (int i = 0; i < m_SkinnedMesh.sharedMesh.blendShapeCount; i++)
            {
                m_SkinnedMesh.SetBlendShapeWeight(i, 0);
            }
        }

        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
        }
    }
    
    private List<BlendshapeFrame> ConvertToBlendshapeFrames(List<float[]> blendshapes)
    {
        string[] blendshapeNames = new string[]
        {
            "eyeBlinking_Left", "eyeBlinking_Right", "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R", "eyeWide_Left", "eyeWide_Right", "jawForward", "jawLeft", "jawRight", "jawOpen", "mouthClosed", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight", "mouthSmile_L", "mouthSmile_R", "mouthFrown_L", "mouthFrown_R", "mouthDimple_L", "mouthDimple_R", "mouthStretch_L", "mouthStretch_R", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPress_L", "mouthPress_R", "mouthLowerDown_L", "mouthLowerDown_R", "mouthUpperUp_L", "mouthUpperUp_R", "browDown_Left", "browDown_Right", "browInnerUp", "browOuterUp_L", "browOuterUp_R", "cheekPuff", "cheekSquint_L", "cheekSquint_R", "noseSneer_L", "noseSneer_R", "tongue_jawOpen", "tongue_jawForward", "tongue_jawLeft", "tongue_jawRight", "tongue_tongueOut",
        };

        List<BlendshapeFrame> frames = new List<BlendshapeFrame>();

        foreach (var blendshapeArray in blendshapes)
        {
            BlendshapeFrame frame = new BlendshapeFrame();
            for (int i = 0; i < ARKIT_BLENDSHAPE_COUNT && i < blendshapeArray.Length; i++)
            {
                frame.values[blendshapeNames[i]] = blendshapeArray[i];
            }
            frames.Add(frame);
        }

        if (blendshapes.Count > 0)
        {
            var lastBlendshapeArray = blendshapes[^1];
            const int FADE_OUT_FRAMES = 20;

            for (int i = 0; i < FADE_OUT_FRAMES; i++)
            {
                var fadeOutFrame = new BlendshapeFrame();
                float fadeFactor = 1.0f - ((float)i / FADE_OUT_FRAMES);

                for (int j = 0; j < ARKIT_BLENDSHAPE_COUNT && j < lastBlendshapeArray.Length; j++)
                {
                    fadeOutFrame.values[blendshapeNames[j]] = lastBlendshapeArray[j] * fadeFactor;
                }
                frames.Add(fadeOutFrame);
            }
        }

        return frames;
    }

    private class BlendshapeFrame
    {
        public Dictionary<string, float> values = new Dictionary<string, float>();
    }
    
    #region Unused Server Communication Code
    
    private class BlendshapesResponse
    {
        public List<float[]> blendshapes { get; set; }
    }

    public async Task WavToBlendshape(AudioClip audioClip)
    {
        byte[] wavData = AudioClipToWav(audioClip);

        using (UnityWebRequest request = new UnityWebRequest(audioToBlendshapeUrl, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(wavData);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "audio/wav");

            try
            {
                var asyncOp = request.SendWebRequest();
                while (!asyncOp.isDone)
                {
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string jsonResponse = request.downloadHandler.text;
                    var response = JsonConvert.DeserializeObject<BlendshapesResponse>(jsonResponse);
                    SetBlendshapeDataAndPlay(response.blendshapes, audioClip);
                }
                else
                {
                    Debug.LogError($"API Error: {request.error}");
                    Debug.LogError($"Response Code: {request.responseCode}");
                    Debug.LogError($"Response: {request.downloadHandler.text}");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Exception occurred: {e.Message}");
            }
        }
    }
    
    private byte[] AudioClipToWav(AudioClip clip)
    {
        float[] samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);

        using (MemoryStream stream = new MemoryStream())
        {
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                writer.Write(Encoding.ASCII.GetBytes("RIFF"));
                writer.Write(0);
                writer.Write(Encoding.ASCII.GetBytes("WAVE"));
                writer.Write(Encoding.ASCII.GetBytes("fmt "));
                writer.Write(16);
                writer.Write((short)1);
                writer.Write((short)clip.channels);
                writer.Write(clip.frequency);
                writer.Write(clip.frequency * clip.channels * 2);
                writer.Write((short)(clip.channels * 2));
                writer.Write((short)16);
                writer.Write(Encoding.ASCII.GetBytes("data"));
                writer.Write(samples.Length * 2);

                foreach (float sample in samples)
                {
                    writer.Write((short)(sample * 32767));
                }

                stream.Position = 4;
                writer.Write((int)(stream.Length - 8));
            }
            return stream.ToArray();
        }
    }
    #endregion
}