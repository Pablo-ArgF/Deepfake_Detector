import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { trackModelDetailsPageView } from '../utils/analytics';

const ModelDetails = () => {
  const { t } = useTranslation();
  const [images, setImages] = useState({
    cnnStructure: '',
    rnnStructure: '',
    cnnGraphs: '',
    rnnGraphs: '',
    cnnConfusion: '',
    rnnConfusion: ''
  });

  useEffect(() => {
    trackModelDetailsPageView();
    loadImages();
  }, []);

  const loadImages = async () => {
    const endpoints = {
      cnnStructure: '/api/model/structure/cnn',
      rnnStructure: '/api/model/structure/rnn',
      cnnGraphs: '/api/model/graphs/cnn',
      rnnGraphs: '/api/model/graphs/rnn',
      cnnConfusion: '/api/model/confussion/matrix/cnn',
      rnnConfusion: '/api/model/confussion/matrix/rnn'
    };

    const loadedImages = {};
    for (const [key, url] of Object.entries(endpoints)) {
      try {
        const response = await fetch(url);
        const text = await response.text();
        loadedImages[key] = 'data:image/png;base64,' + text;
      } catch (error) {
        console.error(`Failed to load image from ${url}:`, error);
      }
    }
    setImages(loadedImages);
  };

  return (
    <div className="flex flex-col gap-12 p-4 md:p-8 max-w-7xl mx-auto">

      {/* Selection Header */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-4">
        <div className="text-center p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-lg">
          <h2 className="text-2xl md:text-3xl font-black text-white mb-2">{t('models.frame_model')}</h2>
          <span className="px-4 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm font-bold border border-blue-500/50 uppercase tracking-widest">Convolutional (CNN)</span>
        </div>
        <div className="text-center p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-lg">
          <h2 className="text-2xl md:text-3xl font-black text-white mb-2">{t('models.seq_model')}</h2>
          <span className="px-4 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-bold border border-purple-500/50 uppercase tracking-widest">Recurrent (RNN)</span>
        </div>
      </div>

      {/* Sections */}
      <div className="flex flex-col gap-16">

        {/* Confusion Matrices */}
        <section>
          <h3 className="text-xl font-bold text-gray-400 mb-6 uppercase tracking-tighter flex items-center gap-2">
            <span className="w-8 h-[1px] bg-gray-700"></span> {t('models.perf_analysis')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ImageCard title={t('models.cnn_matrix')} src={images.cnnConfusion} />
            <ImageCard title={t('models.rnn_matrix')} src={images.rnnConfusion} />
          </div>
        </section>

        {/* Training Graphs */}
        <section>
          <h3 className="text-xl font-bold text-gray-400 mb-6 uppercase tracking-tighter flex items-center gap-2">
            <span className="w-8 h-[1px] bg-gray-700"></span> {t('models.training_progress')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ImageCard title={t('models.cnn_training')} src={images.cnnGraphs} />
            <ImageCard title={t('models.rnn_training')} src={images.rnnGraphs} />
          </div>
        </section>

        {/* Model Structures */}
        <section>
          <h3 className="text-xl font-bold text-gray-400 mb-6 uppercase tracking-tighter flex items-center gap-2">
            <span className="w-8 h-[1px] bg-gray-700"></span> {t('models.arch_design')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ImageCard title={t('models.cnn_arch')} src={images.cnnStructure} />
            <ImageCard title={t('models.rnn_arch')} src={images.rnnStructure} />
          </div>
        </section>

      </div>
    </div>
  );
};

const ImageCard = ({ title, src }) => {
  const { t } = useTranslation();
  return (
    <div className="group flex flex-col bg-gray-800/50 rounded-3xl border border-gray-800 hover:border-blue-500/50 transition-all duration-300 overflow-hidden backdrop-blur-sm">
      <div className="p-4 border-b border-gray-800 bg-gray-800/80">
        <h4 className="text-sm font-bold text-gray-300 uppercase tracking-wide">{title}</h4>
      </div>
      <div className="p-6 flex items-center justify-center bg-gray-900/30">
        {src ? (
          <img
            src={src}
            alt={title}
            className="max-w-full rounded-xl shadow-2xl transition-transform duration-500 group-hover:scale-[1.02]"
          />
        ) : (
          <div className="w-full h-48 bg-gray-800 animate-pulse rounded-xl flex items-center justify-center text-gray-600 italic">
            {t('models.loading')}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelDetails;
