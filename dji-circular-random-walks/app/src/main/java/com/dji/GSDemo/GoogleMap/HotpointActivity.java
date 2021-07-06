package com.dji.GSDemo.GoogleMap;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.location.Location;
import android.location.LocationManager;
import android.os.Build;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.fragment.app.FragmentActivity;

import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.maps.CameraUpdate;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;
import com.google.android.gms.maps.model.PolylineOptions;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentHashMap;

import dji.common.flightcontroller.FlightControllerState;
import dji.common.mission.waypoint.Waypoint;

import dji.common.mission.hotpoint.HotpointMissionState;
import dji.common.mission.hotpoint.HotpointMission;
import dji.common.mission.hotpoint.HotpointMissionEvent;
import dji.common.mission.hotpoint.HotpointHeading;
import dji.common.mission.hotpoint.HotpointStartPoint;

import dji.common.model.LocationCoordinate2D;
import dji.common.useraccount.UserAccountState;
import dji.common.util.CommonCallbacks;
import dji.sdk.base.BaseProduct;
import dji.sdk.flightcontroller.FlightController;
import dji.common.error.DJIError;

import dji.sdk.mission.hotpoint.HotpointMissionOperator;
import dji.sdk.mission.hotpoint.HotpointMissionOperatorListener;

import dji.sdk.products.Aircraft;
import dji.sdk.sdkmanager.DJISDKManager;
import dji.sdk.useraccount.UserAccountManager;

public class HotpointActivity extends FragmentActivity implements View.OnClickListener, GoogleMap.OnMapClickListener, OnMapReadyCallback {

    protected static final String TAG = "GSDemoActivity";

    private GoogleMap gMap;

    private Button locate, add, clear;
    private Button config, upload, start, stop;

    private boolean isAdd = false;

    private double droneLocationLat = 181, droneLocationLng = 181;
    private double prev_droneLocationLat = 181, prev_droneLocationLng = 181;
    private final Map<Integer, Marker> mMarkers = new ConcurrentHashMap<Integer, Marker>();
    private Marker droneMarker = null;

    private float altitude = 100.0f;
    private float mSpeed = 10.0f;

    private final List<Waypoint> waypointList = new ArrayList<>();

    //    public static WaypointMission.Builder waypointMissionBuilder;
    private FlightController mFlightController;
    private HotpointMissionOperator instance;
    private HotpointMission mission;
    private LocationCoordinate2D hotpoint_base;

    // Random walks
    private int curr_orbit = 4;
    private int MAX_ORBIT = 6;
    private int MIN_ORBIT = 2;
    private float ORBIT_WIDTH = 8; // metres
    private float TANGENTIAL_VEL = 5; // m/s

    // JSON
    private JSONArray json_log = new JSONArray();
    private long prev_tsLong = System.currentTimeMillis();

    // Random walk task
    private Timer timer = new Timer();
    private TimerTask randomwalk_task;
    private boolean scheduled = false;

    // Error calc
    private LatLng hotpoint_base_latlng;

//    Handler handler;
//    Runnable periodicUpdate;

    @Override
    protected void onResume() {
        super.onResume();
        initFlightController();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        unregisterReceiver(mReceiver);
        removeListener();
        super.onDestroy();
    }

    /**
     * @Description : RETURN Button RESPONSE FUNCTION
     */
    public void onReturn(View view) {
        Log.d(TAG, "onReturn");
        this.finish();
    }

    private void setResultToToast(final String string) {
        HotpointActivity.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(HotpointActivity.this, string, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void initUI() {
        locate = findViewById(R.id.locate);
        add = findViewById(R.id.add);
        clear = findViewById(R.id.clear);
        config = findViewById(R.id.config);
        upload = findViewById(R.id.upload);
        start = findViewById(R.id.start);
        stop = findViewById(R.id.stop);

        locate.setOnClickListener(this);
        add.setOnClickListener(this);
        clear.setOnClickListener(this);
        config.setOnClickListener(this);
        upload.setOnClickListener(this);
        start.setOnClickListener(this);
        stop.setOnClickListener(this);

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // When the compile and target version is higher than 22, please request the
        // following permissions at runtime to ensure the
        // SDK work well.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.VIBRATE,
                            Manifest.permission.INTERNET, Manifest.permission.ACCESS_WIFI_STATE,
                            Manifest.permission.WAKE_LOCK, Manifest.permission.ACCESS_COARSE_LOCATION,
                            Manifest.permission.ACCESS_NETWORK_STATE, Manifest.permission.ACCESS_FINE_LOCATION,
                            Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS,
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.SYSTEM_ALERT_WINDOW,
                            Manifest.permission.READ_PHONE_STATE,
                    }
                    , 1);
        }

        setContentView(R.layout.activity_waypoint1);

        IntentFilter filter = new IntentFilter();
        filter.addAction(DJIDemoApplication.FLAG_CONNECTION_CHANGE);
        registerReceiver(mReceiver, filter);

        initUI();

        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);

        addListener();

    }

    private void writeToFile(String string){
        FileOutputStream outputStream;

        try {
            outputStream = openFileOutput("output_log.txt", Context.MODE_PRIVATE);
            outputStream.write(string.getBytes());
            outputStream.close();
            setResultToToast("Written to file!");
        } catch (IOException e) {
            setResultToToast("Write failed: " + e);
            e.printStackTrace();
        }
    }

    private void enableLocationSettings() {
        Intent settingsIntent = new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
        startActivity(settingsIntent);
    }

    protected BroadcastReceiver mReceiver = new BroadcastReceiver() {

        @Override
        public void onReceive(Context context, Intent intent) {
            onProductConnectionChange();
        }
    };

    private void onProductConnectionChange() {
        initFlightController();
        loginAccount();
    }

    private void loginAccount() {

        UserAccountManager.getInstance().logIntoDJIUserAccount(this,
                new CommonCallbacks.CompletionCallbackWith<UserAccountState>() {
                    @Override
                    public void onSuccess(final UserAccountState userAccountState) {
                        Log.e(TAG, "Login Success");
                    }

                    @Override
                    public void onFailure(DJIError error) {
                        setResultToToast("Login Error:"
                                + error.getDescription());
                    }
                });
    }


    private void initFlightController() {

        BaseProduct product = DJIDemoApplication.getProductInstance();
        if (product != null && product.isConnected()) {
            if (product instanceof Aircraft) {
                mFlightController = ((Aircraft) product).getFlightController();
            }
        }

        if (mFlightController != null) {
            mFlightController.setStateCallback(new FlightControllerState.Callback() {

                @Override
                public void onUpdate(FlightControllerState djiFlightControllerCurrentState) {
                    droneLocationLat = djiFlightControllerCurrentState.getAircraftLocation().getLatitude();
                    droneLocationLng = djiFlightControllerCurrentState.getAircraftLocation().getLongitude();
                    updateDroneLocation();
                }
            });
        }
    }

    private void addListener() {
        if (getHotpointMissionOperator() != null) {
            getHotpointMissionOperator().addListener(eventNotificationListener);
        }
    }

    private void removeListener() {
        if (getHotpointMissionOperator() != null) {
            getHotpointMissionOperator().removeListener(eventNotificationListener);
        }
    }

    private HotpointMissionOperatorListener eventNotificationListener = new HotpointMissionOperatorListener() {
        @Override
        public void onExecutionUpdate(@NonNull HotpointMissionEvent hotpointMissionEvent) {

        }

        @Override
        public void onExecutionStart() {

        }

        @Override
        public void onExecutionFinish(@Nullable final DJIError error) {
            setResultToToast("Execution finished: " + (error == null ? "Success!" : error.getDescription()));
        }
    };

    public HotpointMissionOperator getHotpointMissionOperator() {
        if (instance == null) {
            if (DJISDKManager.getInstance().getMissionControl() != null) {
                instance = DJISDKManager.getInstance().getMissionControl().getHotpointMissionOperator();
            }
        }
        return instance;
    }

    private void setUpMap() {
        gMap.setOnMapClickListener(this);// add the listener for click for amap object
    }

    @Override
    public void onMapClick(LatLng point) {
        if (isAdd) {
            markHotpoint(point);
            hotpoint_base_latlng = point;
            hotpoint_base = new LocationCoordinate2D(point.latitude, point.longitude);
            enableDisableAdd(); // Toggle the button after adding hotpoint home loc
        } else {
            setResultToToast("Cannot Add Hotpoint");
        }
    }

    public Location getCurrentLocation() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            setResultToToast("Location permissions not set");
            return null;
        }

        LocationManager locationManager = (LocationManager) getApplicationContext().getSystemService(LOCATION_SERVICE);
        List<String> providers = locationManager.getProviders(true);
        Location bestLocation = null;
        for (String provider : providers) {

            Location temp_location = locationManager.getLastKnownLocation(provider);
            if (temp_location == null) {
                continue;
            }
            if (bestLocation == null || temp_location.getAccuracy() < bestLocation.getAccuracy()) {
                // Found best last known location: %s", l);
                bestLocation = temp_location;
            }
        }
        return bestLocation;
    }

    public static boolean checkGpsCoordination(double latitude, double longitude) {
        return (latitude > -90 && latitude < 90 && longitude > -180 && longitude < 180) && (latitude != 0f && longitude != 0f);
    }


    // Update the drone location based on states from MCU.
    private void updateDroneLocation() {
        LatLng pos = new LatLng(droneLocationLat, droneLocationLng);

        // Add to json
        long tsLong = System.currentTimeMillis();
        String ts = "" + tsLong;

        if ((tsLong - prev_tsLong >= 250) && scheduled){
            float[] results = new float[1];
            if (!Double.isNaN(droneLocationLat) && !Double.isNaN(droneLocationLng) &&
                    !Double.isNaN(hotpoint_base.getLatitude()) && !Double.isNaN(hotpoint_base.getLongitude())){
                Location.distanceBetween(pos.latitude, pos.longitude, hotpoint_base.getLatitude(), hotpoint_base.getLongitude(), results);
            }

            JSONObject json_temp = new JSONObject();
            try {
                json_temp.put("time", "" + ts);
                json_temp.put("lat", "" + droneLocationLat);
                json_temp.put("lng", "" + droneLocationLng);
                json_temp.put("radius", "" + curr_orbit * ORBIT_WIDTH);
                json_temp.put("dist", "" + results[0]);
            } catch (JSONException e) {
            }

            json_log.put(json_temp);
            prev_tsLong = tsLong;
        }

        //Create MarkerOptions object
        final MarkerOptions markerOptions = new MarkerOptions();
        markerOptions.position(pos);
        markerOptions.icon(BitmapDescriptorFactory.fromResource(R.drawable.aircraft));

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (droneMarker != null) {
                    droneMarker.remove();
                }

                if (checkGpsCoordination(droneLocationLat, droneLocationLng)) {
                    // Drawing a line on the path
                    LatLng prev_pos = new LatLng(prev_droneLocationLat, prev_droneLocationLng);

                    gMap.addPolyline(new PolylineOptions()
                            .add(prev_pos, pos)
                            .width(2)
                            .color(Color.CYAN)
                    );

                    prev_droneLocationLat = droneLocationLat;
                    prev_droneLocationLng = droneLocationLng;

                    droneMarker = gMap.addMarker(markerOptions);
                }
            }
        });
    }

    private void markHotpoint(LatLng point) {
        //Create MarkerOptions object
        MarkerOptions markerOptions = new MarkerOptions();
        markerOptions.position(point);
        markerOptions.icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_BLUE));
        Marker marker = gMap.addMarker(markerOptions);
        mMarkers.put(mMarkers.size(), marker);
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.locate: {
                updateDroneLocation();
                cameraUpdate(); // Locate the drone's place
                break;
            }
            case R.id.add: {
                enableDisableAdd();
                break;
            }
            case R.id.clear: {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        gMap.clear();
                    }
                });
                updateDroneLocation();
                break;
            }
            case R.id.config: {
                showSettingDialog();
                break;
            }
            case R.id.upload: {
                uploadHotPointMission();
                break;
            }
            case R.id.start: {
                 startHotpointMission();

                if (!scheduled){
                    scheduled = true;
                    randomwalk_task  = new TimerTask() {
                        @Override
                        public void run() {
                            doRandomWalk();
                        }
                    };
                    timer.schedule(randomwalk_task, 20*1000, 20*1000);
                }
                break;
            }
            case R.id.stop: {
                stopHotpointMission();
                if (scheduled){
                    randomwalk_task.cancel();
                    scheduled = false;
                    writeToFile(json_log.toString());
                }

                break;
            }
            default:
                break;
        }
    }

    // random walk markov chain
    private int getMarkovNextState(){
        double P_i_to_i_minus_1;
        double P_i_to_i_plus_1;
        double P_i_to_i;

        if (curr_orbit == MIN_ORBIT){
            P_i_to_i_minus_1 = 0;
            P_i_to_i_plus_1 = 0.5;
            P_i_to_i = 0.5;
        }
        else if (curr_orbit == MAX_ORBIT) {
            P_i_to_i_minus_1 = 1-1/(double)(2*curr_orbit);
            P_i_to_i_plus_1 = 0;
            P_i_to_i = 1/(double)(2*curr_orbit);
        }
        else {
            P_i_to_i_minus_1 = (curr_orbit-1)/(double)(2*curr_orbit);
            P_i_to_i_plus_1 = 0.5;
            P_i_to_i = 1/(double)(2*curr_orbit);
        }


        double rand = Math.random(); // generate a random number in [0,1]
        int state;
        if (0 <= rand && rand < P_i_to_i_minus_1){
            state = curr_orbit-1;
        }
        else if (P_i_to_i_minus_1 <= rand && rand < P_i_to_i_minus_1 + P_i_to_i){
            state = curr_orbit;
        }
        else{
            state = curr_orbit+1;
        }

        return state;
    }

    // Every 10 seconds
    private void doRandomWalk(){
        if (!(
                (getHotpointMissionOperator().getCurrentState() == HotpointMissionState.INITIAL_PHASE) ||
                (getHotpointMissionOperator().getCurrentState() == HotpointMissionState.EXECUTING) ||
                (getHotpointMissionOperator().getCurrentState() == HotpointMissionState.EXECUTION_PAUSED)
             )){
            setResultToToast("Unable to do random walk");
            return;
        }

//        float max_ang_vel = HotpointMissionOperator.getMaxAngularVelocityForRadius(curr_orbit*ORBIT_WIDTH,
//                new dji.common.util.DJICommonCallbacks.DJICompletionCallbackWith  CommonCallbacks.CompletionCallbackWith<Float>(float a){
//                    @Override
//                    public void onSuccess(Float aFloat) {
//
//                    }
//
//                    @Override
//                    public void onFailure(DJIError djiError) {
//
//                    }
//                });

        curr_orbit = getMarkovNextState();
        float ang_vel = (float)(TANGENTIAL_VEL/(curr_orbit*ORBIT_WIDTH) * 360/(2*Math.PI));

        getHotpointMissionOperator().setAngularVelocity(ang_vel, error -> setResultToToast("New ang_vel " + ang_vel));
        getHotpointMissionOperator().setRadius(curr_orbit*ORBIT_WIDTH, error -> setResultToToast("New orbit " + curr_orbit));
    }

    private void cameraUpdate() {
        LatLng pos = new LatLng(droneLocationLat, droneLocationLng);
        float zoomlevel = (float) 18.0;
        CameraUpdate cu = CameraUpdateFactory.newLatLngZoom(pos, zoomlevel);
        gMap.moveCamera(cu);
    }

    private void enableDisableAdd() {
        if (!isAdd) {
            isAdd = true;
            add.setText("Exit");
        } else {
            isAdd = false;
            add.setText("Add");
        }
    }

    private void showSettingDialog() {
        LinearLayout wayPointSettings = (LinearLayout) getLayoutInflater().inflate(R.layout.dialog_waypointsetting, null);

        final TextView wpAltitude_TV = wayPointSettings.findViewById(R.id.altitude);
        RadioGroup speed_RG = wayPointSettings.findViewById(R.id.speed);
        RadioGroup actionAfterFinished_RG = wayPointSettings.findViewById(R.id.actionAfterFinished);
        RadioGroup heading_RG = wayPointSettings.findViewById(R.id.heading);

        speed_RG.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {

            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.lowSpeed) {
                    mSpeed = 3.0f;
                } else if (checkedId == R.id.MidSpeed) {
                    mSpeed = 5.0f;
                } else if (checkedId == R.id.HighSpeed) {
                    mSpeed = 10.0f;
                }
            }

        });

        new AlertDialog.Builder(this)
                .setTitle("")
                .setView(wayPointSettings)
                .setPositiveButton("Finish", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {

                        String altitudeString = wpAltitude_TV.getText().toString();
                        altitude = Integer.parseInt(nulltoIntegerDefalt(altitudeString));
                        Log.e(TAG, "altitude " + altitude);
                        Log.e(TAG, "speed " + mSpeed);
                    }

                })
                .setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        dialog.cancel();
                    }

                })
                .create()
                .show();
    }

    String nulltoIntegerDefalt(String value) {
        if (!isIntValue(value)) value = "0";
        return value;
    }

    boolean isIntValue(String val) {
        try {
            val = val.replace(" ", "");
            Integer.parseInt(val);
        } catch (Exception e) {
            return false;
        }
        return true;
    }


    private void uploadHotPointMission() {
        setResultToToast("Not implemented for hotpoint mission");

    }

    private void startHotpointMission() {
        double altitude = 20; // Altitude of flight (m)
        double radius = curr_orbit * ORBIT_WIDTH; // Radius of circle (m)
        float angularVelocity = (float)(TANGENTIAL_VEL/(curr_orbit*ORBIT_WIDTH) * 360/(2*Math.PI));; // Angular velocity (deg/s)
        boolean isClockwise = false; // Direction of rotation
        HotpointStartPoint startPoint = HotpointStartPoint.NEAREST; // Start from [N/S/E/W/Nearest] point on circle?
        HotpointHeading heading = HotpointHeading.ALONG_CIRCLE_LOOKING_FORWARDS; // Where the drone is facing
                                                                                 // (towards, away, forward, backward, original, useRemote)

        if (mission == null) {
            mission = new HotpointMission(hotpoint_base, altitude, radius,
                    angularVelocity, isClockwise, startPoint, heading);
        }

        if (instance.getCurrentState() == HotpointMissionState.READY_TO_EXECUTE) {
            getHotpointMissionOperator().startMission(mission, new CommonCallbacks.CompletionCallback() {
                @Override
                public void onResult(DJIError error) {
                    setResultToToast("Mission Start: " + (error == null ? "Successfully" : error.getDescription()));
                }
            });
        } else {
            setResultToToast("Cannot start mission..." + instance.getCurrentState());
        }
    }

    private void stopHotpointMission() {
        getHotpointMissionOperator().stop(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onResult(DJIError error) {
                setResultToToast("Mission Stop: " + (error == null ? "Successfully" : error.getDescription()));
            }
        });
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        setResultToToast("Starting the map");
        if (gMap == null) {
            gMap = googleMap;
            setUpMap();
        }

        Location currloc = getCurrentLocation();
        LatLng here;
        if (currloc == null){
            // Marking the ece building
            setResultToToast("Could not get location. Going to the ece building...");
            here = new LatLng(40.4287257, -86.9119606);
        }
        else {
            here = new LatLng(currloc.getLatitude(), currloc.getLongitude());
        }

        gMap.addMarker(new MarkerOptions().position(here).title("Marker at your loc"));
        gMap.moveCamera(CameraUpdateFactory.newLatLng(here));
        gMap.moveCamera((CameraUpdateFactory.zoomTo(17)));
    }
}
