#!/bin/sh
gst-launch-1.0 \
  rtpbin name=rtpbin \
    udpsrc port=10051 caps="application/x-rtp, media=video, encoding-name=VP8, clock-rate=90000" ! rtpbin.recv_rtp_sink_1 \
      rtpbin. ! rtpvp8depay ! vp8dec ! videoconvert ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=480,height=320,framerate=5/1 ! \
      tensor_converter ! tensor_filter framework=pytorch model=pytorch_animae_ganv2_paprika_gpu.pt input=3:480:320:1 inputtype=uint8 output=3:480:320:1 outputtype=uint8 accelerator=true:gpu ! \
      tensor_decoder mode=direct_video ! \
      videoscale ! videorate ! videoconvert ! timeoverlay ! \
      vp8enc error-resilient=1 ! \
      rtpvp8pay ! udpsink host=127.0.0.1 port=5004 sync=false


      # videoconvert ! fpsdisplaysink sync=false\

# gst-launch-1.0 \
#   rtpbin name=rtpbin \
#     udpsrc port=10050 caps="application/x-rtp, media=audio, encoding-name=OPUS, clock-rate=48000" ! rtpbin.recv_rtp_sink_0 \
#     udpsrc port=10051 caps="application/x-rtp, media=video, encoding-name=VP8, clock-rate=90000" ! rtpbin.recv_rtp_sink_1 \
#       # rtpbin. ! rtpvp8depay ! vp8dec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! fakesink
#       rtpbin. ! rtpvp8depay ! vp8dec ! autovideosink \
#       rtpbin. ! rtpopusdepay ! queue ! opusdec ! pulsesink
