import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { trackAboutPageView } from '../utils/analytics';

const About = () => {
    const { t } = useTranslation();

    useEffect(() => {
        trackAboutPageView();
    }, []);


    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] p-8 max-w-6xl mx-auto">
            <h1 className="text-4xl md:text-5xl font-black text-white mb-16 text-center tracking-tight">
                {t('about.title')}
            </h1>

            <div className="flex flex-col lg:flex-row items-center lg:items-center gap-12 bg-gray-800/50 p-8 md:p-12 rounded-[3rem] border border-gray-700 shadow-2xl backdrop-blur-sm">

                {/* Developer Profile Card */}
                <div className="flex flex-col items-center shrink-0">
                    <div className="relative group p-2 bg-gradient-to-tr from-blue-500 to-purple-600 rounded-[2.5rem] shadow-xl">
                        <img
                            className="h-64 md:h-80 w-auto rounded-[2rem] object-cover group-hover:scale-[1.01] transition-transform duration-500"
                            src="./face.JPEG"
                            alt={t('about.name')}
                        />
                    </div>

                    <div className="flex gap-4 mt-8">
                        <SocialButton
                            icon="./github.png"
                            url="https://github.com/Pablo-ArgF"
                            label="GitHub"
                        />
                        <SocialButton
                            icon="./linkedin.png"
                            url="https://www.linkedin.com/in/pablo-argallero/"
                            label="LinkedIn"
                        />
                    </div>
                </div>

                {/* Info Text */}
                <div className="flex flex-col gap-6 max-w-xl text-justify">
                    <div className="space-y-2">
                        <h2 className="text-2xl md:text-3xl font-black text-white">{t('about.name')}</h2>
                        <a
                            href="https://pabloaf.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400 font-bold uppercase tracking-widest text-sm hover:text-blue-300 transition-colors"
                        >
                            {t('about.role')}
                        </a>
                    </div>

                    <div className="space-y-4 text-gray-300 text-lg leading-relaxed">
                        <p>
                            {t('about.p1_start')} <a href="https://hdl.handle.net/10651/74464" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 font-bold underline underline-offset-4 decoration-blue-500/30 hover:decoration-blue-400 transition-all">{t('about.p1_link')}</a>, {t('about.p1_end')}
                        </p>
                        <p>
                            {t('about.p2')}
                        </p>
                        <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                            <li><span className="text-blue-400 font-semibold">Grad-CAM:</span> {t('about.li1')}</li>
                            <li><span className="text-purple-400 font-semibold">Heatmaps:</span> {t('about.li2')}</li>
                            <li><span className="text-white font-semibold">Enhanced UI:</span> {t('about.li3')}</li>
                        </ul>
                        <p>
                            {t('about.p3')}
                        </p>
                        <hr className="border-gray-700 my-6" />
                        <p className="text-sm italic text-gray-400">
                            {t('about.collab')}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

const SocialButton = ({ icon, url, label }) => (
    <button
        onClick={() => window.open(url, '_blank')}
        className="p-3 bg-gray-900/50 border border-gray-700 rounded-2xl hover:bg-gray-700 hover:scale-110 transition-all duration-300"
        title={label}
    >
        <img src={icon} alt={label} className="w-8 h-8 rounded-lg" />
    </button>
);

export default About;
