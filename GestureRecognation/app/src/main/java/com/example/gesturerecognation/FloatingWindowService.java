package com.example.gesturerecognation;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

public class FloatingWindowService extends Service {

    private WindowManager windowManager;
    private View floatingView1;
    private View floatingView2;
    private GestureRecorder gestureRecorder;
    private Handler handler;
    private boolean isLongPress = false;
    private static final String CHANNEL_ID = "FloatingWindowServiceChannel";
    private int savedX = 0;
    private int savedY = 100;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        gestureRecorder = new GestureRecorder(this);
        handler = new Handler();

        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        floatingView1 = LayoutInflater.from(this).inflate(R.layout.floating_window, null);
        floatingView2 = LayoutInflater.from(this).inflate(R.layout.floating_window_2, null);

        int layoutFlag;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            layoutFlag = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY;
        } else {
            layoutFlag = WindowManager.LayoutParams.TYPE_PHONE;
        }
        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                layoutFlag,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);

        params.gravity = Gravity.TOP | Gravity.LEFT;
        params.x = savedX;
        params.y = savedY;

        setupFloatingView1(params);
        setupFloatingView2(params);
    }

    private void setupFloatingView1(final WindowManager.LayoutParams params) {
        Button buttonRecording = floatingView1.findViewById(R.id.buttonRecording);
        ImageView buttonSwitch = floatingView1.findViewById(R.id.buttonSwitch);
        View backgroundCircle = floatingView1.findViewById(R.id.backgroundCircle);

        if (floatingView1.getParent() == null) {
            windowManager.addView(floatingView1, params);
        }

        backgroundCircle.setOnTouchListener(new View.OnTouchListener() {
            private int initialX;
            private int initialY;
            private float initialTouchX;
            private float initialTouchY;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = params.x;
                        initialY = params.y;
                        initialTouchX = event.getRawX();
                        initialTouchY = event.getRawY();
                        handler.postDelayed(longPressRunnable, ViewConfiguration.getLongPressTimeout());
                        return true;
                    case MotionEvent.ACTION_UP:
                        handler.removeCallbacks(longPressRunnable);
                        if (isLongPress) {
                            isLongPress = false;
                            gestureRecorder.stopRecording();
                        } else {
                            snapToEdge(params, floatingView1);
                        }
                        savePosition(params.x, params.y);
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        handler.removeCallbacks(longPressRunnable);
                        params.x = initialX + (int) (event.getRawX() - initialTouchX);
                        params.y = initialY + (int) (event.getRawY() - initialTouchY);
                        windowManager.updateViewLayout(floatingView1, params);
                        return true;
                }
                return false;
            }

            private Runnable longPressRunnable = new Runnable() {
                @Override
                public void run() {
                    isLongPress = true;
                    gestureRecorder.startRecording();
                }
            };
        });

        buttonRecording.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        handler.postDelayed(longPressRunnable, ViewConfiguration.getLongPressTimeout());
                        return true;
                    case MotionEvent.ACTION_UP:
                        handler.removeCallbacks(longPressRunnable);
                        if (isLongPress) {
                            isLongPress = false;
                            gestureRecorder.stopRecording();
                        }
                        return true;
                }
                return false;
            }

            private Runnable longPressRunnable = new Runnable() {
                @Override
                public void run() {
                    isLongPress = true;
                    gestureRecorder.startRecording();
                }
            };
        });

        buttonSwitch.setOnClickListener(v -> {
            if (floatingView1.getParent() != null) {
                windowManager.removeView(floatingView1);
            }
            if (floatingView2.getParent() == null) {
                windowManager.addView(floatingView2, params);
            }
            floatingView2.setVisibility(View.VISIBLE);
        });
    }

    private void setupFloatingView2(final WindowManager.LayoutParams params) {
        ImageView icon = floatingView2.findViewById(R.id.icon);

        floatingView2.setOnTouchListener(new View.OnTouchListener() {
            private int initialX;
            private int initialY;
            private float initialTouchX;
            private float initialTouchY;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = params.x;
                        initialY = params.y;
                        initialTouchX = event.getRawX();
                        initialTouchY = event.getRawY();
                        return true;
                    case MotionEvent.ACTION_UP:
                        snapToEdge(params, floatingView2);
                        savePosition(params.x, params.y);
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        params.x = initialX + (int) (event.getRawX() - initialTouchX);
                        params.y = initialY + (int) (event.getRawY() - initialTouchY);
                        windowManager.updateViewLayout(floatingView2, params);
                        return true;
                }
                return false;
            }
        });

        icon.setOnClickListener(v -> {
            if (floatingView2.getParent() != null) {
                windowManager.removeView(floatingView2);
            }
            if (floatingView1.getParent() == null) {
                windowManager.addView(floatingView1, params);
            }
            floatingView1.setVisibility(View.VISIBLE);
        });
    }

    private void snapToEdge(WindowManager.LayoutParams params, View view) {
        int middle = getResources().getDisplayMetrics().widthPixels / 2;
        if (params.x >= middle) {
            params.x = getResources().getDisplayMetrics().widthPixels - view.getWidth();
        } else {
            params.x = 0;
        }
        windowManager.updateViewLayout(view, params);
    }

    private void savePosition(int x, int y) {
        savedX = x;
        savedY = y;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            String action = intent.getStringExtra("action");
            if ("show".equals(action)) {
                if (floatingView1.getParent() == null) {
                    WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                            WindowManager.LayoutParams.WRAP_CONTENT,
                            WindowManager.LayoutParams.WRAP_CONTENT,
                            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O ?
                                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY :
                                    WindowManager.LayoutParams.TYPE_PHONE,
                            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                            PixelFormat.TRANSLUCENT);
                    params.gravity = Gravity.TOP | Gravity.LEFT;
                    params.x = savedX;
                    params.y = savedY;
                    windowManager.addView(floatingView1, params);
                }
                floatingView1.setVisibility(View.VISIBLE);
                floatingView2.setVisibility(View.GONE);
            } else if ("hide".equals(action)) {
                floatingView1.setVisibility(View.GONE);
                floatingView2.setVisibility(View.GONE);
            }
        }
        return START_STICKY;
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(
                    CHANNEL_ID,
                    "Floating Window Service Channel",
                    NotificationManager.IMPORTANCE_DEFAULT
            );
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(serviceChannel);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (floatingView1 != null && floatingView1.getParent() != null) {
            windowManager.removeView(floatingView1);
        }
        if (floatingView2 != null && floatingView2.getParent() != null) {
            windowManager.removeView(floatingView2);
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
