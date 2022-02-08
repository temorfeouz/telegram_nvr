Telegram NVR

Application analyze video stream, detects movement and if min brightness of video more than threshold video will be sent to telegram

Example command for run application:
python webstreaming.py --ip %IP FOR WEB% --port 8080 --stream "%CAMERA STREAM%" --min-brightness %MIN BRIGHTNESS FOR RECORD VIDEO% --telegram-token "%TELEGRAM TOKEN%" --telegram-chat %CHAT GROUP OR CHAT ID FOR SEND VIDEOS% 
