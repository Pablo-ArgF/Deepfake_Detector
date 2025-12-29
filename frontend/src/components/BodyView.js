import React, { useState } from 'react';
import { IoMdVideocam } from 'react-icons/io';
import { HiArrowLeft } from 'react-icons/hi';
import CNNVideoDashboard from './CNNVideoDashboard';
import RNNVideoDashboard from './RNNVideoDashboard';

const BodyView = () => {
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [RNNdata, setRNNData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(1);
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [RNNloading, setRNNLoading] = useState(true);
  const [useRNN, setUseRNN] = useState(false);

  const handleVideoUpload = async (event) => {
    setError('');
    const file = event.target.files[0];
    if (!file) {
      setError('Please select a video file.');
      return;
    }
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('video', file);

      const controller = new AbortController();
      var timeoutId = setTimeout(() => controller.abort(), 600000);

      fetch('/api/predict', {
        method: 'POST',
        body: formData,
        headers: {
          'enctype': 'multipart/form-data'
        },
        signal: controller.signal
      }).then(async response => {
        if (!response.ok) {
          setError('An error occurred while processing the video, please try again.');
          setLoading(false);
          setVideoUploaded(false);
          setData(null);
          return;
        }
        var CNNdata;
        try {
          CNNdata = await response.json();
        } catch (error) {
          setError('An error occurred while processing the video, please try again.');
          setLoading(false);
          setVideoUploaded(false);
          setData(null);
          return;
        }
        setData(CNNdata);
        setLoading(false);
        setVideoUploaded(true);
        clearTimeout(timeoutId);

        var timeoutIdRNN = setTimeout(() => controller.abort(), 600000);
        //TODO Uncomment when CNN is implemented
        // fetch('/api/predict/sequences', {
        //   method: 'POST',
        //   body: formData,
        //   headers: {
        //     'enctype': 'multipart/form-data'
        //   },
        //   signal: controller.signal
        // }).then(async RNNresponse => {
        //       try {
        //         const RNNTmpdata = await RNNresponse.json();
        //         setRNNData(RNNTmpdata);
        //         setRNNLoading(false);
        //       } catch (error) {
        //         setError('An error occurred while processing the video sequences.');
        //         setLoading(false);
        //         setVideoUploaded(false);
        //       }
        //       clearTimeout(timeoutIdRNN);
        //     }).catch(error => {
        //       setError('An error occurred while processing the video sequences.');
        //       setLoading(false);
        //       setVideoUploaded(false);
        //     });
      });
    } catch (error) {
      setError('Error predicting DeepFakes: ' + error);
      setLoading(false);
      setVideoUploaded(false);
      setData(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 md:p-8">
      {videoUploaded && (
        <div className="flex flex-col md:flex-row items-center justify-between mb-8 gap-4">
          <div className="flex items-center gap-4 w-full md:w-auto">
            <button
              onClick={() => {
                setVideoUploaded(false);
                setData(null);
                setUseRNN(false);
                setRNNData(null);
                setTimeout(() => {
                  setLoading(false);
                }, 1000);
              }}
              className="p-3 bg-gray-800 hover:bg-gray-700 rounded-xl transition-colors shadow-md border border-gray-700"
              aria-label="Back to main menu"
            >
              <HiArrowLeft className="text-2xl" />
            </button>
            <h1 className="text-2xl md:text-4xl font-extrabold tracking-tight">
              {useRNN ? 'Sequence Analysis (RNN)' : 'Frame-by-Frame Analysis (CNN)'}
            </h1>
          </div>
          <button
            onClick={() => setUseRNN(!useRNN)}
            className="w-full md:w-64 py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl transition-all shadow-lg transform hover:scale-105"
          >
            {useRNN ? 'Analyze frame by frame' : 'Analyze sequences'}
          </button>
        </div>
      )}

      {videoUploaded ? (
        useRNN ? (
          <RNNVideoDashboard
            setData={setRNNData}
            setLoading={setRNNLoading}
            loading={RNNloading}
            data={RNNdata}
            setSelectedIndex={setSelectedIndex}
            selectedIndex={selectedIndex}
          />
        ) : (
          <CNNVideoDashboard
            setVideoUploaded={setVideoUploaded}
            setData={setData}
            setLoading={setLoading}
            data={data}
            setSelectedIndex={setSelectedIndex}
            selectedIndex={selectedIndex}
          />
        )
      ) : (
        <div className="flex flex-col items-center max-w-4xl mx-auto py-12 px-6 bg-gray-800 rounded-3xl shadow-2xl border border-gray-700 mt-10">
          <h1 className="text-4xl md:text-6xl font-black mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            DeepFake Detection
          </h1>
          <p className="text-lg text-gray-300 text-justify leading-relaxed mb-8 max-w-2xl px-4">
            The increasing use of artificial intelligence has enabled identity impersonation through <b>DeepFakes</b>, synthetic content generated by AI algorithms that combine and overlay existing images and videos to create a new one.<br /><br />
            The realism achieved using current DeepFake algorithms poses a <b>risk to society</b>. From 'fake news' to identity theft, AI-generated content makes it increasingly difficult to distinguish real from fake.<br /><br />
            This <b>DeepFake detection tool</b> aims to help identify synthetic material through predictive model analysis.
          </p>

          {error && (
            <div className="mb-6 p-4 bg-red-900/30 border border-red-500 text-red-200 rounded-xl text-center">
              {error}
            </div>
          )}

          <div className="flex flex-col items-center gap-6">
            <input
              type="file"
              id="videoInput"
              accept="video/mp4"
              onChange={handleVideoUpload}
              className="hidden"
            />
            <label
              htmlFor="videoInput"
              className="group flex items-center gap-3 py-4 px-8 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xl rounded-2xl cursor-pointer transition-all shadow-xl hover:shadow-blue-500/20 transform hover:-translate-y-1"
            >
              <IoMdVideocam className="text-2xl transition-transform group-hover:scale-110" />
              Upload a video
            </label>

            {loading && (
              <div className="mt-8 flex flex-col items-center gap-4">
                <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                <p className="text-blue-400 font-medium animate-pulse">Processing video...</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default BodyView;
