import React from 'react';

const About = () => {
    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] p-8 max-w-6xl mx-auto">
            <h1 className="text-4xl md:text-5xl font-black text-white mb-16 text-center tracking-tight">
                Project Information
            </h1>

            <div className="flex flex-col lg:flex-row items-center lg:items-center gap-12 bg-gray-800/50 p-8 md:p-12 rounded-[3rem] border border-gray-700 shadow-2xl backdrop-blur-sm">

                {/* Developer Profile Card */}
                <div className="flex flex-col items-center shrink-0">
                    <div className="relative group p-2 bg-gradient-to-tr from-blue-500 to-purple-600 rounded-[2.5rem] shadow-xl">
                        <img
                            className="h-64 md:h-80 w-auto rounded-[2rem] object-cover group-hover:scale-[1.01] transition-transform duration-500"
                            src="./face.JPEG"
                            alt="Pablo Argallero Fernández"
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
                <div className="flex flex-col gap-6 max-w-xl">
                    <div className="space-y-2">
                        <h2 className="text-2xl md:text-3xl font-black text-white">Pablo Argallero Fernández</h2>
                        <p className="text-blue-400 font-bold uppercase tracking-widest text-sm">Computer Engineering Student</p>
                    </div>

                    <div className="space-y-4 text-gray-300 text-lg leading-relaxed text-auto">
                        <p>
                            Student at the <span className="text-white font-semibold">University of Oviedo</span>,
                            developed this project as his Bachelor's Thesis.
                        </p>
                        <p>
                            The project explores advanced <span className="text-white font-semibold">DeepFake detection</span>
                            techniques in videos, utilizing state-of-the-art Deep Learning models:
                            <span className="text-blue-400 font-bold"> CNN</span> (Convolutional Neural Networks) and
                            <span className="text-purple-400 font-bold"> RNN</span> (Recurrent Neural Networks).
                        </p>
                        <p>
                            This research encompassed extensive image preprocessing, model architecture design,
                            training pipelines, and exhaustive result analysis, all controlled through
                            this modern web interface.
                        </p>
                        <hr className="border-gray-700 my-6" />
                        <p className="text-sm italic text-gray-400">
                            Supervised by professor <span className="text-gray-200 font-medium NOT-italic">Cristian González García</span>.
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
