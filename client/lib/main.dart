import 'dart:io';
import 'package:ffmpeg_kit_flutter_full_gpl/ffmpeg_session.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_vlc_player/flutter_vlc_player.dart';
import 'package:ffmpeg_kit_flutter_full_gpl/ffmpeg_kit.dart';
import 'package:network_info_plus/network_info_plus.dart';
import 'package:permission_handler/permission_handler.dart';

enum SourceType { file, camera, network }

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MaterialApp(home: VideoStreamPage()));
}

class VideoStreamPage extends StatefulWidget {
  const VideoStreamPage({super.key});

  @override
  _VideoStreamPageState createState() => _VideoStreamPageState();
}

class _VideoStreamPageState extends State<VideoStreamPage> {
  String? rtspUrl;
  String? currentSource;
  VlcPlayerController? videoController;
  FFmpegSession? ffmpegExecutionId;
  List<CameraDescription> cameras = [];

  @override
  void initState() {
    super.initState();
    initCameras();
    requestPermissions();
  }

  Future<void> initCameras() async {
    cameras = await availableCameras();
  }

  Future<bool> requestPermissions() async {
    final statuses =
        await [
          Permission.camera,
          Permission.microphone,
          Permission.storage,
        ].request();
    return statuses[Permission.camera]?.isGranted == true &&
        statuses[Permission.microphone]?.isGranted == true &&
        statuses[Permission.storage]?.isGranted == true;
  }

  Future<void> selectVideoSource() async {
    final type = await showDialog<SourceType>(
      context: context,
      builder:
          (context) => SimpleDialog(
            title: Text('选择视频源'),
            children: [
              _buildSourceOption('文件', SourceType.file),
              _buildSourceOption('摄像头', SourceType.camera),
              _buildSourceOption('网络源', SourceType.network),
            ],
          ),
    );

    if (type != null) {
      switch (type) {
        case SourceType.file:
          await handleFileSource();
          break;
        case SourceType.camera:
          await handleCameraSource();
          break;
        case SourceType.network:
          await handleNetworkSource();
          break;
      }
    }
  }

  Widget _buildSourceOption(String text, SourceType type) {
    return SimpleDialogOption(
      onPressed: () => Navigator.pop(context, type),
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 12),
        child: Text(text),
      ),
    );
  }

  Future<void> handleFileSource() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.video);
    if (result != null) {
      startFFmpegProcess(
        input: result.files.single.path!,
        type: SourceType.file,
      );
    }
  }

  Future<void> handleCameraSource() async {
    if (cameras.isEmpty) return;

    final selected = await showDialog<CameraDescription>(
      context: context,
      builder:
          (context) => SimpleDialog(
            title: Text('选择摄像头'),
            children:
                cameras
                    .map(
                      (camera) => SimpleDialogOption(
                        onPressed: () => Navigator.pop(context, camera),
                        child: Text(
                          camera.lensDirection == CameraLensDirection.front
                              ? '前置摄像头'
                              : '后置摄像头',
                        ),
                      ),
                    )
                    .toList(),
          ),
    );

    if (selected != null) {
      startFFmpegProcess(
        input: cameras.indexOf(selected).toString(),
        type: SourceType.camera,
      );
    }
  }

  Future<void> handleNetworkSource() async {
    final controller = TextEditingController();
    final url = await showDialog<String>(
      context: context,
      builder:
          (context) => AlertDialog(
            title: Text('输入网络地址'),
            content: TextField(controller: controller),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, controller.text),
                child: Text('确定'),
              ),
            ],
          ),
    );

    if (url != null && url.isNotEmpty) {
      startFFmpegProcess(input: url, type: SourceType.network);
    }
  }

  Future<void> startFFmpegProcess({
    required String input,
    required SourceType type,
  }) async {
    // 停止现有进程
   
    await ffmpegExecutionId?.cancel();


    final ip = await NetworkInfo().getWifiIP() ?? 'localhost';
    final output = 'rtsp://$ip:8554/stream';

    String command = '';
    switch (type) {
      case SourceType.file:
        command = '-re -i "$input" -c:v copy -f rtsp "$output"';
        break;
      case SourceType.camera:
        if (Platform.isAndroid) {
          command =
              '-f android_camera -camera_index $input -video_size 1280x720 -i "" '
              '-c:v libx264 -preset ultrafast -f rtsp "$output"';
        } else {
          command =
              '-f avfoundation -video_device_index $input -i "" -c:v libx264 -preset ultrafast -f rtsp "$output"';
        }
        break;
      case SourceType.network:
        command = '-i "$input" -c:v copy -f rtsp "$output"';
        break;
    }

    // 启动FFmpeg进程
    ffmpegExecutionId = await FFmpegKit.executeAsync(command, (session) async {
      final code = await session.getReturnCode();
      print('FFmpeg process exited with code: $code');
    });

    // 更新UI
    setState(() {
      rtspUrl = output;
      currentSource = _getSourceName(type);
      videoController?.dispose();
      videoController = VlcPlayerController.network(
        output,
        options: VlcPlayerOptions(),
      );
    });
  }

  String _getSourceName(SourceType type) {
    switch (type) {
      case SourceType.file:
        return '文件';
      case SourceType.camera:
        return '摄像头';
      case SourceType.network:
        return '网络源';
    }
  }

  @override
  void dispose() {
    videoController?.dispose();
    ffmpegExecutionId?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('视频流传输')),
      body: Column(
        children: [
          ElevatedButton(onPressed: selectVideoSource, child: Text('选择视频源')),
          Padding(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('当前源: ${currentSource ?? "未选择"}'),
                Text('RTSP地址: ${rtspUrl ?? "未启动"}'),
              ],
            ),
          ),
          Expanded(
            child:
                rtspUrl != null
                    ? VlcPlayer(
                      controller: videoController!,
                      aspectRatio: 16 / 9,
                    )
                    : Center(child: Text('请先选择视频源')),
          ),
        ],
      ),
    );
  }
}
