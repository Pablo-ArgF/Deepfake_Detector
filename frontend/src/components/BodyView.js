import React, { useState, useEffect } from 'react';
import { IoMdVideocam } from 'react-icons/io';
import { HiArrowLeft, HiAcademicCap } from 'react-icons/hi';
import { useNavigate, useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CNNVideoDashboard from './CNNVideoDashboard';
import RNNVideoDashboard from './RNNVideoDashboard';
import { FaLinkedin, FaExternalLinkAlt } from 'react-icons/fa';
import {
  trackDemoAccess,
  trackCNNDemoAccess,
  trackRNNDemoAccess,
  trackVideoUploadSubmit,
  trackVideoAnalysisSuccess,
  trackVideoAnalysisError,
  trackModelSwitch,
  trackTutorialClick
} from '../utils/analytics';


const UploadModal = ({ isOpen, onClose, onUpload, password, setPassword, error, loading }) => {
  const { t } = useTranslation();
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-300">
      <div className="bg-gray-800 border border-gray-700 rounded-3xl p-8 max-w-lg w-full shadow-2xl overflow-hidden relative">
        {loading && (
          <div className="absolute inset-0 z-10 bg-gray-800/80 flex flex-col items-center justify-center backdrop-blur-sm transition-all">
            <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mb-4"></div>
            <p className="text-blue-400 font-bold animate-pulse text-xl uppercase tracking-widest">{t('body.processing')}</p>
            <p className="text-gray-400 text-sm mt-2">{t('body.processing_desc')}</p>
          </div>
        )}

        <h2 className="text-3xl font-black mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
          {t('body.upload_title')}
        </h2>
        <p className="text-gray-300 mb-6 leading-relaxed">
          {t('body.upload_desc')}
        </p>

        <div className="flex flex-col gap-4 mb-8">
          <a
            href="https://www.linkedin.com/in/pablo-argallero/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 p-4 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/50 rounded-2xl transition-all group"
          >
            <FaLinkedin className="text-2xl text-blue-400 group-hover:scale-110 transition-transform" />
            <span className="text-blue-100 font-bold uppercase tracking-wider">LinkedIn Profile</span>
          </a>
          <a
            href="https://pabloaf.com/#contact"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 p-4 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/50 rounded-2xl transition-all group"
          >
            <FaExternalLinkAlt className="text-xl text-purple-400 group-hover:scale-110 transition-transform" />
            <span className="text-purple-100 font-bold uppercase tracking-wider">Contact at pabloaf.com</span>
          </a>
        </div>

        <div className="flex flex-col gap-3">
          <label className="text-sm font-semibold text-gray-400">{t('body.enter_password')}</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
            className="bg-gray-900 border border-gray-600 rounded-2xl px-6 py-4 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-white font-bold disabled:opacity-50"
            placeholder="••••••••"
          />
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-900/30 border border-red-500/50 text-red-200 rounded-xl text-sm text-center">
            {error}
          </div>
        )}

        <div className="flex gap-4 mt-8">
          <button
            onClick={onClose}
            disabled={loading}
            className="flex-1 py-4 px-6 bg-gray-700 hover:bg-gray-600 text-white font-bold rounded-2xl transition-all disabled:opacity-50"
          >
            {t('body.cancel')}
          </button>
          <button
            onClick={onUpload}
            disabled={loading}
            className="flex-1 py-4 px-6 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-2xl transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50"
          >
            {t('body.submit')}
          </button>
        </div>
      </div>
    </div>
  );
};

const BodyView = () => {
  const navigate = useNavigate();
  const { uuid, type } = useParams();
  const { t } = useTranslation();

  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [RNNdata, setRNNData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [RNNloading, setRNNLoading] = useState(true);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [password, setPassword] = useState('');
  const [tempFile, setTempFile] = useState(null);
  const [runTutorial, setRunTutorial] = useState(false);

  // Effect to load data if UUID is in URL
  useEffect(() => {
    if (uuid) {
      setLoading(true);
      const isDemo = uuid === 'demo';
      let url = isDemo ? (type === 'rnn' ? '/demo/sequences/demo/demo.json' : '/demo/frames/demo/demo.json') : `/api/results/${uuid}`;

      fetch(url)
        .then(res => {
          if (!res.ok) throw new Error('Result not found');
          return res.json();
        })
        .then(resultData => {
          const finalData = isDemo ? { ...resultData, isDemo: true, uuid: 'demo' } : resultData;
          setSelectedIndex(0);

          // Track demo access
          if (isDemo) {
            if (type === 'rnn') {
              trackRNNDemoAccess();
            } else {
              trackCNNDemoAccess();
            }
          }

          if (finalData.type === 'cnn') {
            setData(finalData);
            setVideoUploaded(true);
          } else {
            setRNNData(finalData);
            setRNNLoading(false);
            setVideoUploaded(true);
          }
          setLoading(false);
        })
        .catch(err => {
          setError(t('body.error_not_found'));
          setLoading(false);
          navigate('/');
        });
    } else {
      // Reset state if no UUID
      setVideoUploaded(false);
      setData(null);
      setRNNData(null);
      setSelectedIndex(0);
    }
  }, [uuid, navigate, t]);

  const handleVideoUpload = async (event) => {
    setError('');
    const file = event.target.files[0];
    if (!file) return;
    setTempFile(file);
    setIsModalOpen(true);
  };

  const executeUpload = async () => {
    if (!tempFile) return;
    setError('');

    try {
      setLoading(true);

      // Track video upload submission
      trackVideoUploadSubmit();

      const formData = new FormData();
      formData.append('video', tempFile);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 600000);

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'X-Upload-Password': password
        },
        body: formData,
        signal: controller.signal
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || t('body.processing_error');

        // Track CNN analysis error
        trackVideoAnalysisError('cnn', errorMsg);

        throw new Error(errorMsg);
      }

      const CNNData = await response.json();
      clearTimeout(timeoutId);

      // Track successful CNN analysis
      trackVideoAnalysisSuccess('cnn');

      setIsModalOpen(false);
      setTempFile(null);
      setPassword('');

      // Navigate to CNN view after prediction
      navigate(`/${CNNData.uuid}/cnn`);

      // Start RNN prediction in background using the SAME UUID
      // We pass the uuid as a query parameter so the backend reuses the folder/video
      fetch(`/api/predict/sequences?uuid=${CNNData.uuid}`, {
        method: 'POST',
        headers: {
          'X-Upload-Password': password
        }
        // No body needed if uuid is provided as it reuses the file!
      }).then(async RNNresponse => {
        if (!RNNresponse.ok) {
          trackVideoAnalysisError('rnn', 'RNN analysis failed');
          throw new Error('RNN analysis failed');
        }
        const RNNTmpdata = await RNNresponse.json();

        // Track successful RNN analysis
        trackVideoAnalysisSuccess('rnn');

        setRNNData(RNNTmpdata);
        setRNNLoading(false);
      }).catch(err => {
        console.error('RNN error:', err);
        trackVideoAnalysisError('rnn', err.message);
        setRNNLoading(false); // Stop spinner even if it fails
      });

    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleDemo = async () => {
    try {
      setLoading(true);

      // Track demo access
      trackDemoAccess();

      const response = await fetch('/demo/frames/demo/demo.json');
      if (!response.ok) throw new Error('Demo data not found');
      const demoData = await response.json();

      // Inject demo flag and dummy UUID for demo images to work if needed
      const preparedData = {
        ...demoData,
        isDemo: true,
        uuid: 'demo'
      };

      setSelectedIndex(0);
      setData(preparedData);
      setVideoUploaded(true);
      setLoading(false);
      navigate('/demo/cnn');
    } catch (err) {
      setError('Error loading demo: ' + err.message);
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
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight flex items-center gap-3 main-dashboard-header">
              {isRNN ? t('body.seq_analysis_title') : t('body.frame_analysis_title')}
              <button
                onClick={() => {
                  trackTutorialClick(isRNN ? 'rnn' : 'cnn');
                  setRunTutorial(true);
                }}
                className={`p-2 ${isRNN ? 'bg-purple-600 hover:bg-purple-700' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded-xl transition-all shadow-lg flex items-center gap-2`}
                title={t('common.tutorial')}
              >
                <HiAcademicCap className="text-xl" />
                <span className="text-[10px] font-black uppercase tracking-widest hidden sm:inline">{t('common.tutorial')}</span>
              </button>
            </h1>
          </div>
          <button
            onClick={() => {
              const from = isRNN ? 'rnn' : 'cnn';
              const to = isRNN ? 'cnn' : 'rnn';
              trackModelSwitch(from, to);
              navigate(`/${uuid}/${to}`);
            }}
            className={`w-full md:w-64 py-3 px-6 ${isRNN ? 'bg-blue-600 hover:bg-blue-700' : 'bg-purple-600 hover:bg-purple-700'} text-white font-bold rounded-xl transition-all shadow-lg transform hover:scale-105 uppercase tracking-tight text-xs`}
          >
            {isRNN ? t('body.analyze_frames') : t('body.analyze_sequences')}
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
            runTutorial={runTutorial}
            setRunTutorial={setRunTutorial}
          />
        ) : (
          <CNNVideoDashboard
            setVideoUploaded={setVideoUploaded}
            setData={setData}
            setLoading={setLoading}
            data={data}
            setSelectedIndex={setSelectedIndex}
            selectedIndex={selectedIndex}
            runTutorial={runTutorial}
            setRunTutorial={setRunTutorial}
          />
        )
      ) : (
        <div className="flex flex-col items-center max-w-4xl mx-auto py-12 px-6 bg-gray-800 rounded-3xl shadow-2xl border border-gray-700 mt-10">
          <h1 className="text-4xl md:text-6xl font-black mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            {t('body.hero_title')}
          </h1>

          <div className="mb-8 w-48 md:w-64 rounded-3xl overflow-hidden shadow-2xl border border-gray-700">
            <img
              src="/deepfake-paper.jpg"
              alt="Deepfake Detection Project"
              className="w-full h-auto"
            />
          </div>
          <p className="text-lg text-gray-300 text-justify leading-relaxed mb-8 max-w-2xl px-4">
            {t('body.intro_p1')}<br /><br />
            {t('body.intro_p2')}<br /><br />
            {t('body.intro_p3')}
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
              {t('body.upload_btn')}
            </label>

            <button
              onClick={handleDemo}
              className="group flex items-center gap-3 py-4 px-8 bg-purple-600 hover:bg-purple-700 text-white font-bold text-xl rounded-2xl cursor-pointer transition-all shadow-xl hover:shadow-purple-500/20 transform hover:-translate-y-1"
            >
              🚀 {t('body.demo_btn')}
            </button>

            {loading && !isModalOpen && (
              <div className="mt-8 flex flex-col items-center gap-4">
                <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                <p className="text-blue-400 font-medium animate-pulse">{t('body.processing')}</p>
              </div>
            )}
          </div>
        </div>
      )}

      <UploadModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setTempFile(null);
          setPassword('');
          setError('');
        }}
        onUpload={executeUpload}
        password={password}
        setPassword={setPassword}
        error={error}
        loading={loading}
      />
    </div>
  );
};

export default BodyView;
