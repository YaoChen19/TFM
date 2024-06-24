package com.example.tfmapplication;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Handler;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RecordActivity extends AppCompatActivity {
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private List<Float> sensorDataX;
    private List<Float> sensorDataY;
    private List<Float> sensorDataZ;
    private String gestureLabel;
    private TextView recordingStatus;
    private boolean isRecording = false;
    private Handler handler = new Handler();
    private int counter = 0;
    private TextView counterView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record);

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorDataX = new ArrayList<>();
        sensorDataY = new ArrayList<>();
        sensorDataZ = new ArrayList<>();

        gestureLabel = getIntent().getStringExtra("gesture_label");
        recordingStatus = findViewById(R.id.recording_status);
        counterView = findViewById(R.id.counter_view);

        Button buttonRecording = findViewById(R.id.button_recording);
        buttonRecording.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        handler.post(startRecordingRunnable);
                        return true;
                    case MotionEvent.ACTION_UP:
                        handler.post(stopRecordingRunnable);
                        return true;
                }
                return false;
            }
        });
    }

    private final Runnable startRecordingRunnable = new Runnable() {
        @Override
        public void run() {
            startRecording();
        }
    };

    private final Runnable stopRecordingRunnable = new Runnable() {
        @Override
        public void run() {
            stopRecording();
        }
    };

    private void startRecording() {
        isRecording = true;
        recordingStatus.setVisibility(View.VISIBLE);
        sensorDataX.clear();
        sensorDataY.clear();
        sensorDataZ.clear();
        sensorManager.registerListener(sensorEventListener, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show();
    }

    private void stopRecording() {
        isRecording = false;
        recordingStatus.setVisibility(View.GONE);
        sensorManager.unregisterListener(sensorEventListener);
        saveData(sensorDataX, sensorDataY, sensorDataZ, gestureLabel);
        Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show();
        updateCounter();
    }

    private void saveData(List<Float> dataX, List<Float> dataY, List<Float> dataZ, String label) {
        try (FileWriter writer = new FileWriter(getExternalFilesDir(null) + "/gestures.csv", true)) {
            writer.append(label).append(",");
            writer.append("\"").append(dataX.toString()).append("\",");
            writer.append("\"").append(dataY.toString()).append("\",");
            writer.append("\"").append(dataZ.toString()).append("\"\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void updateCounter() {
        counter++;
        if (counter >= 20) {
            counter = 0;
        }
        counterView.setText(String.valueOf(counter));
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
