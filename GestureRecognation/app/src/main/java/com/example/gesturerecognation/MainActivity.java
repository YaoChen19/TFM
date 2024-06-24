package com.example.gesturerecognation;

import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int SETTINGS_REQUEST_CODE = 1;
    private static final int OVERLAY_PERMISSION_REQUEST_CODE = 2;
    private GestureRecorder gestureRecorder;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        gestureRecorder = new GestureRecorder(this);
        resultText = findViewById(R.id.resultText);

        // 检查并请求悬浮窗权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(this)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE);
            } else {
                // 启动浮动窗口服务
                startFloatingWindowService();
            }
        } else {
            // 启动浮动窗口服务
            startFloatingWindowService();
        }

        Button buttonRecording = findViewById(R.id.buttonRecording);
        ImageButton buttonSettings = findViewById(R.id.buttonSettings);

        buttonRecording.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        gestureRecorder.startRecording();
                        return true;
                    case MotionEvent.ACTION_UP:
                        gestureRecorder.stopRecording();
                        updateResultText(gestureRecorder.getDetectedGesture());
                        return true;
                }
                return false;
            }
        });

        buttonSettings.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
            startActivityForResult(intent, SETTINGS_REQUEST_CODE);
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            gestureRecorder.setGestureMappings(
                    data.getStringExtra("gestureA"),
                    data.getStringExtra("gestureU"),
                    data.getStringExtra("gestureD"),
                    data.getStringExtra("gestureL")
            );
        } else if (requestCode == OVERLAY_PERMISSION_REQUEST_CODE) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (Settings.canDrawOverlays(this)) {
                    // 启动浮动窗口服务
                    startFloatingWindowService();
                } else {
                    Toast.makeText(this, "悬浮窗权限未授予", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        hideFloatingWindow();
    }

    @Override
    protected void onPause() {
        super.onPause();
        showFloatingWindow();
    }

    private void startFloatingWindowService() {
        Intent intent = new Intent(this, FloatingWindowService.class);
        startService(intent);
    }

    private void showFloatingWindow() {
        Intent intent = new Intent(this, FloatingWindowService.class);
        intent.putExtra("action", "show");
        startService(intent);
    }

    private void hideFloatingWindow() {
        Intent intent = new Intent(this, FloatingWindowService.class);
        intent.putExtra("action", "hide");
        startService(intent);
    }

    private void updateResultText(String gesture) {
        resultText.setText("Detected Gesture: " + gesture);
    }
}
