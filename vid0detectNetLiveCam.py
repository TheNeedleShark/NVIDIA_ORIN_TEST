import cv2
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, videoOutput

# Replace with your yt-dlp output stream URL
stream_url = '''
https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1752533608/ei/CDZ1aNeCH-yQsfIPjqyLsQY/ip/68.65.166.186/id/w3LjpFhySTg.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag
%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr1---sn-jxopj-nh4e.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AY1jyLNytR2NJ2N0mBKHP80zFEgJqx5pLoUgHl92V95lIMQbNF3Q07MRrx7pW4TN_83QKM
j-lwg6rSGi/spc/l3OVKQIK4-qKkm61VDXZd2QmFWO78c-VIN07m8xo4zHBJuZib1NIYmzq__LBx5sI_ruCm1xarnQ/vprv/1/playlist_type/DVR/initcwndbps/1863750/met/1752512010,/mh/9x/mm/44/mn/sn-jxopj-nh4e/ms/lva/mv/m/mvi/1/pl/20/rms/lva,lva/dover/11/paci
ng/0/keepalive/yes/fexp/51355912/mt/1752511729/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRQIgLms3JIlvmHqIA5VTYdgH-
5J1F2J_ihnvyseXdIX6YzMCIQDxKui1CzosMlBsK6LVTaWlfp8ECbckSCiXKEPTozWHpQ%3D%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRQIgTeiW7c_IPajkNCpT5-Q8pH1rf7nzQNfrtWJzv6AjZfgCIQCTPtyo0F8FYlvE1zN0ANvEEWcy2J8vjzwwpj2OCg9KqQ%3D%3D/playlist/index.m3u8"
'''
cap = cv2.VideoCapture(stream_url)
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
display = videoOutput("display://0")  # or "my_output.mp4", "rtp://...", etc.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cuda_img = cudaFromNumpy(frame)
    detections = net.Detect(cuda_img)
    display.Render(cuda_img)
    display.SetStatus("Object Detection | {:.0f} FPS".format(net.GetNetworkFPS()))


