package com.example.tfmapplication;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonGestureA = findViewById(R.id.button_gesture_u);
        Button buttonGestureU = findViewById(R.id.button_gesture_u);
        Button buttonGestureD = findViewById(R.id.button_gesture_d);
        Button buttonGestureL = findViewById(R.id.button_gesture_l);
        Button buttonGestureR = findViewById(R.id.button_gesture_r);
        Button buttonGestureX = findViewById(R.id.button_gesture_x);

        View.OnClickListener gestureButtonClickListener = v -> {
            String gesture = ((Button) v).getText().toString();
            Intent intent = new Intent(MainActivity.this, RecordActivity.class);
            intent.putExtra("gesture_label", gesture);
            startActivity(intent);
        };

        buttonGestureA.setOnClickListener(gestureButtonClickListener);
        buttonGestureU.setOnClickListener(gestureButtonClickListener);
        buttonGestureD.setOnClickListener(gestureButtonClickListener);
        buttonGestureL.setOnClickListener(gestureButtonClickListener);
        buttonGestureR.setOnClickListener(gestureButtonClickListener);
        buttonGestureX.setOnClickListener(gestureButtonClickListener);
    }
}
