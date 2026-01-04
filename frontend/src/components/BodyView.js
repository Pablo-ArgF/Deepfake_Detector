import React, { useState, useEffect } from 'react';
import { IoMdVideocam } from 'react-icons/io';
import { HiArrowLeft } from 'react-icons/hi';
import { useNavigate, useParams } from 'react-router-dom';
import CNNVideoDashboard from './CNNVideoDashboard';
import RNNVideoDashboard from './RNNVideoDashboard';

const BodyView = () => {
  const navigate = useNavigate();
  const { uuid, type } = useParams();

  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [RNNdata, setRNNData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(1);
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [RNNloading, setRNNLoading] = useState(true);

  // Effect to load data if UUID is in URL
  useEffect(() => {
    if (uuid) {
      setLoading(true);
      fetch(`/api/results/${uuid}`)
        .then(res => {
          if (!res.ok) throw new Error('Result not found');
          return res.json();
        })
        .then(resultData => {
          if (resultData.type === 'cnn') {
            setData(resultData);
            setVideoUploaded(true);
          } else {
            setRNNData(resultData);
            setRNNLoading(false);
            setVideoUploaded(true);
          }
          setLoading(false);
        })
        .catch(err => {
          setError('Analysis not found or expired.');
          setLoading(false);
          navigate('/');
        });
    } else {
      // Reset state if no UUID
      setVideoUploaded(false);
      setData(null);
      setRNNData(null);
    }
  }, [uuid, navigate]);

  const handleVideoUpload = async (event) => {
    setError('');
    const file = event.target.files[0];
    if (!file) return;

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('video', file);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 600000);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      if (!response.ok) throw new Error('Error processing video');

      const CNNData = await response.json();
      clearTimeout(timeoutId);

      // Navigate to CNN view after prediction
      navigate(`/${CNNData.uuid}/cnn`);

      // Start RNN prediction in background if needed, or wait for it
      // For now, let's just trigger the sequences API and navigate when it returns or handle it in RNN view
      fetch('/api/predict/sequences', {
        method: 'POST',
        body: formData,
      }).then(async RNNresponse => {
        const RNNTmpdata = await RNNresponse.json();
        setRNNData(RNNTmpdata);
        setRNNLoading(false);
      }).catch(err => console.error('RNN error:', err));

    } catch (err) {
      setError('Error predicting DeepFakes: ' + err.message);
      setLoading(false);
    }
  };

  const isRNN = type === 'rnn';

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 md:p-8">
      {videoUploaded && (
        <div className="flex flex-col md:flex-row items-center justify-between mb-8 gap-4">
          <div className="flex items-center gap-4 w-full md:w-auto">
            <button
              onClick={() => navigate('/')}
              className="p-3 bg-gray-800 hover:bg-gray-700 rounded-xl transition-colors shadow-md border border-gray-700"
              aria-label="Back to main menu"
            >
              <HiArrowLeft className="text-2xl" />
            </button>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">
              {isRNN ? 'Sequence Analysis (RNN)' : 'Frame-by-Frame Analysis (CNN)'}
            </h1>
          </div>
          <button
            onClick={() => navigate(`/${uuid}/${isRNN ? 'cnn' : 'rnn'}`)}
            className="w-full md:w-64 py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl transition-all shadow-lg transform hover:scale-105"
          >
            {isRNN ? 'Analyze frame by frame' : 'Analyze sequences'}
          </button>
        </div>
      )}

      {videoUploaded ? (
        isRNN ? (
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

          <div className="mb-8 w-48 md:w-64 rounded-3xl overflow-hidden shadow-2xl border border-gray-700">
            <img
              src="/deepfake-paper.jpg"
              alt="Deepfake Detection Project"
              className="w-full h-auto"
            />
          </div>
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
