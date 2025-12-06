import React from 'react';

export default function VideoPlayer() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white p-6">
      <div className="w-full max-w-4xl">
        <h1 className="text-2xl font-semibold mb-4">Recording Preview</h1>
        <div className="bg-black rounded-lg overflow-hidden">
          {/*
            Place your MP4 at `frontend/public/videos/recording.mp4`.
            The dev server will serve it at `/videos/recording.mp4`.
          */}
          <video controls style={{ width: '100%', height: 'auto' }}>
            <source src="/videos/recording.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
        <p className="text-sm text-gray-300 mt-3">If the video does not load, make sure you copied the MP4 into <code>/frontend/public/videos/recording.mp4</code>.</p>
      </div>
    </div>
  );
}
