package com.example.gesturerecognation;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class SettingsActivity extends AppCompatActivity {
    private static final String PREFS_NAME = "GesturePrefs";
    private Spinner spinnerA, spinnerU, spinnerD, spinnerL;
    private SharedPreferences sharedPreferences;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        spinnerA = findViewById(R.id.spinnerA);
        spinnerU = findViewById(R.id.spinnerU);
        spinnerD = findViewById(R.id.spinnerD);
        spinnerL = findViewById(R.id.spinnerL);
        Button buttonSave = findViewById(R.id.buttonSave);

        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.gestures_array, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        spinnerA.setAdapter(adapter);
        spinnerU.setAdapter(adapter);
        spinnerD.setAdapter(adapter);
        spinnerL.setAdapter(adapter);

        sharedPreferences = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);

        loadSettings();

        buttonSave.setOnClickListener(v -> {
            saveSettings();
            finish();
        });
    }

    private void loadSettings() {
        spinnerA.setSelection(getIndex(spinnerA, sharedPreferences.getString("gestureA", "led1")));
        spinnerU.setSelection(getIndex(spinnerU, sharedPreferences.getString("gestureU", "led2")));
        spinnerD.setSelection(getIndex(spinnerD, sharedPreferences.getString("gestureD", "led3")));
        spinnerL.setSelection(getIndex(spinnerL, sharedPreferences.getString("gestureL", "led4")));
    }

    private int getIndex(Spinner spinner, String value) {
        int index = 0;
        for (int i = 0; i < spinner.getCount(); i++) {
            if (spinner.getItemAtPosition(i).toString().equalsIgnoreCase(value)) {
                index = i;
                break;
            }
        }
        return index;
    }

    private void saveSettings() {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("gestureA", spinnerA.getSelectedItem().toString());
        editor.putString("gestureU", spinnerU.getSelectedItem().toString());
        editor.putString("gestureD", spinnerD.getSelectedItem().toString());
        editor.putString("gestureL", spinnerL.getSelectedItem().toString());
        editor.apply();

        Intent resultIntent = new Intent();
        resultIntent.putExtra("gestureA", spinnerA.getSelectedItem().toString());
        resultIntent.putExtra("gestureU", spinnerU.getSelectedItem().toString());
        resultIntent.putExtra("gestureD", spinnerD.getSelectedItem().toString());
        resultIntent.putExtra("gestureL", spinnerL.getSelectedItem().toString());
        setResult(RESULT_OK, resultIntent);
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
}
