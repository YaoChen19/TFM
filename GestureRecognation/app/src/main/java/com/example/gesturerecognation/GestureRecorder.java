package com.example.gesturerecognation;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.Toast;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class GestureRecorder {
    private static final String TAG = "GestureRecorder";
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private List<Float> sensorDataX;
    private List<Float> sensorDataY;
    private List<Float> sensorDataZ;
    private GestureClassifier gestureClassifier;
    private boolean isRecording = false;
    private String serverUrl = "http://192.168.1.132:3000";
    private TextToSpeech textToSpeech;
    private Vibrator vibrator;

    private String gestureALed;
    private String gestureULed;
    private String gestureDLed;
    private String gestureLLed;
    private String detectedGesture = "Unknown gesture";
    private Context context;
    private SharedPreferences sharedPreferences;

    private static final String PREFS_NAME = "GesturePrefs";
    private static final String GESTURE_A_KEY = "gestureA";
    private static final String GESTURE_U_KEY = "gestureU";
    private static final String GESTURE_D_KEY = "gestureD";
    private static final String GESTURE_L_KEY = "gestureL";

    public GestureRecorder(Context context) {
        this.context = context;
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorDataX = new ArrayList<>();
        sensorDataY = new ArrayList<>();
        sensorDataZ = new ArrayList<>();

        vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
        textToSpeech = new TextToSpeech(context, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
            }
        });

        sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        loadGestureMappings();

        try {
            gestureClassifier = new GestureClassifier(context);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(context, "Failed to initialize gesture classifier", Toast.LENGTH_LONG).show();
        }
    }

    public void setGestureMappings(String gestureA, String gestureU, String gestureD, String gestureL) {
        this.gestureALed = gestureA;
        this.gestureULed = gestureU;
        this.gestureDLed = gestureD;
        this.gestureLLed = gestureL;
        saveGestureMappings();
    }

    private void saveGestureMappings() {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString(GESTURE_A_KEY, gestureALed);
        editor.putString(GESTURE_U_KEY, gestureULed);
        editor.putString(GESTURE_D_KEY, gestureDLed);
        editor.putString(GESTURE_L_KEY, gestureLLed);
        editor.apply();
    }

    private void loadGestureMappings() {
        gestureALed = sharedPreferences.getString(GESTURE_A_KEY, "led1");
        gestureULed = sharedPreferences.getString(GESTURE_U_KEY, "led2");
        gestureDLed = sharedPreferences.getString(GESTURE_D_KEY, "led3");
        gestureLLed = sharedPreferences.getString(GESTURE_L_KEY, "led4");
    }

    public void startRecording() {
        if (!isRecording) {
            isRecording = true;
            sensorDataX.clear();
            sensorDataY.clear();
            sensorDataZ.clear();
            sensorManager.registerListener(sensorEventListener, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            Toast.makeText(context, "Recording started", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "Recording started");
        }
    }

    public void stopRecording() {
        if (isRecording) {
            isRecording = false;
            sensorManager.unregisterListener(sensorEventListener);

            try {
                // Converting data to float[][][]
                int maxLength = 83; // Maximum length expected by the model
                float[][][] gestureData = new float[1][maxLength][3];
                for (int i = 0; i < maxLength; i++) {
                    gestureData[0][i][0] = i < sensorDataX.size() ? sensorDataX.get(i) : 0;
                    gestureData[0][i][1] = i < sensorDataY.size() ? sensorDataY.get(i) : 0;
                    gestureData[0][i][2] = i < sensorDataZ.size() ? sensorDataZ.get(i) : 0;
                }

                // 标准化数据
                gestureData = standardizeData(gestureData);

                // 调用分类器
                String result = gestureClassifier.classify(gestureData);

                // 生成详细消息
                detectedGesture = getDetailedResult(result);
                Log.d(TAG, "Gesture classified: " + detectedGesture);

                // 语音输出结果
                textToSpeech.speak(detectedGesture, TextToSpeech.QUEUE_FLUSH, null, null);

                // 震动
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE));
                } else {
                    vibrator.vibrate(500);
                }

                // 根据分类结果发送相应的命令
                switch (result) {
                    case "Gesture A":
                        sendCommandToServer(gestureALed);
                        break;
                    case "Gesture U":
                        sendCommandToServer(gestureULed);
                        break;
                    case "Gesture D":
                        sendCommandToServer(gestureDLed);
                        break;
                    case "Gesture L":
                        sendCommandToServer(gestureLLed);
                        break;
                    default:
                        Log.d(TAG, "Unknown gesture: " + result);
                        break;
                }

            } catch (Exception e) {
                Log.e(TAG, "Error during stopRecording", e);
                Toast.makeText(context, "Error during stopRecording: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
    }

    public String getDetectedGesture() {
        return detectedGesture;
    }

    private float[][][] standardizeData(float[][][] data) {
        // Assuming the data has been normalized between 0 and 1
        // Adjust according to the normalization method used to train the model
        // If not normalized, this step is not needed
        return data;
    }

    private void sendCommandToServer(String led) {
        new Thread(() -> {
            try {
                URL url = new URL(serverUrl + "/control");
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json");
                connection.setDoOutput(true);

                String jsonInputString = "{\"led\": \"" + led + "\"}";
                Log.d(TAG, "Sending command to server: " + jsonInputString);

                try (OutputStream os = connection.getOutputStream()) {
                    byte[] input = jsonInputString.getBytes("utf-8");
                    os.write(input, 0, input.length);
                }

                int responseCode = connection.getResponseCode();
                Log.d(TAG, "Response Code: " + responseCode);

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    Log.d(TAG, "Command sent successfully");
                } else {
                    Log.d(TAG, "Failed to send command. Response Code: " + responseCode);
                }

                connection.disconnect();
            } catch (IOException e) {
                Log.e(TAG, "Error sending command to server", e);
            }
        }).start();
    }

    private String getDetailedResult(String gesture) {
        switch (gesture) {
            case "Gesture A":
                return "Gesture A detected";
            case "Gesture U":
                return "Gesture U detected";
            case "Gesture D":
                return "Gesture D detected";
            case "Gesture L":
                return "Gesture L detected";
            default:
                return "Unknown gesture detected";
        }
    }

    private final SensorEventListener sensorEventListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            if (isRecording && event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                sensorDataX.add(event.values[0]);
                sensorDataY.add(event.values[1]);
                sensorDataZ.add(event.values[2]);
            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            // Do nothing
        }
    };
}
