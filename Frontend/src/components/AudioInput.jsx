import React, { useState, useEffect, useRef } from 'react';

const AudioInput = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const uploadAudio = async (audioBlob, fileName = 'recording.wav') => {
    const formData = new FormData();
    formData.append('audio', audioBlob, fileName);

    try {
      const response = await fetch('http://your-backend-url.com/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('Audio uploaded successfully');
      } else {
        console.error('Failed to upload audio');
      }
    } catch (error) {
      console.error('Error uploading audio:', error);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      uploadAudio(file, file.name);
    }
  };

  useEffect(() => {
    if (isRecording) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          mediaRecorderRef.current = new MediaRecorder(stream);

          mediaRecorderRef.current.ondataavailable = (event) => {
            audioChunksRef.current.push(event.data);
          };

          mediaRecorderRef.current.onstop = () => {
            const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
            setAudioUrl(URL.createObjectURL(audioBlob));
            audioChunksRef.current = []; // Clear chunks for the next recording
            uploadAudio(audioBlob); // Upload the audio to the backend
          };

          mediaRecorderRef.current.start();
        })
        .catch((error) => {
          console.error('Error accessing microphone', error);
        });
    } else {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    }

    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, [isRecording]);

  return (
    <div>
      <button onClick={() => setIsRecording((prev) => !prev)}>
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </button>
      {audioUrl && <audio controls src={audioUrl} />}
      <div>
        <input type="file" accept="audio/*" onChange={handleFileSelect} />
      </div>
    </div>
  );
};

export default AudioInput;
