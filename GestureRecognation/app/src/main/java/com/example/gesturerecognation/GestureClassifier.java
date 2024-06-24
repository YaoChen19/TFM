package com.example.gesturerecognation;

import android.content.Context;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

public class GestureClassifier {
    private Interpreter tflite;
    private final Map<Integer, String> labelMap;

    public GestureClassifier(Context context) throws IOException {
        MappedByteBuffer tfliteModel = loadModelFile(context, "gesture_modelCNNNORX.tflite");
        tflite = new Interpreter(tfliteModel);

        // Add Tag Mapping
        labelMap = new HashMap<>();
        labelMap.put(0, "Gesture A");
        labelMap.put(1, "Gesture D");
        labelMap.put(2, "Gesture L");
        labelMap.put(3, "Gesture U");
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        FileInputStream inputStream = new FileInputStream(context.getAssets().openFd(modelPath).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = context.getAssets().openFd(modelPath).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelPath).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public String classify(float[][][] inputData) {
        // Create input tensor
        TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, 83, 3}, org.tensorflow.lite.DataType.FLOAT32);
        float[] inputArray = new float[83 * 3]; // 75 个时间步，每个有 3 个特征
        for (int i = 0; i < 83; i++) {
            inputArray[i * 3] = inputData[0][i][0];
            inputArray[i * 3 + 1] = inputData[0][i][1];
            inputArray[i * 3 + 2] = inputData[0][i][2];
        }
        inputBuffer.loadArray(inputArray);

        // Create output tensor
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 4}, org.tensorflow.lite.DataType.FLOAT32);

        // running inference
        tflite.run(inputBuffer.getBuffer(), outputBuffer.getBuffer());

        // Get results and return labels
        float[] output = outputBuffer.getFloatArray();
        int maxIndex = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[maxIndex]) {
                maxIndex = i;
            }
        }
        return labelMap.get(maxIndex);
    }
}