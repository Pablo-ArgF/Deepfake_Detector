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
                <div className="flex flex-col gap-6 max-w-xl text-justify">
                    <div className="space-y-2">
                        <h2 className="text-2xl md:text-3xl font-black text-white">Pablo Argallero Fernández</h2>
                        <a
                            href="https://pabloaf.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400 font-bold uppercase tracking-widest text-sm hover:text-blue-300 transition-colors"
                        >
                            Software Engineer · Pabloaf.com
                        </a>
                    </div>

                    <div className="space-y-4 text-gray-300 text-lg leading-relaxed">
                        <p>
                            This research initiative represents the academic evolution of the <a href="https://hdl.handle.net/10651/74464" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 font-bold underline underline-offset-4 decoration-blue-500/30 hover:decoration-blue-400 transition-all">Deepfake Detection Final Degree Project</a>, shifting focus from pure detection accuracy to interpretability and transparency in AI systems. The primary objective is to contribute to the scientific community by producing a detailed paper on Explainable AI (XAI) applied to forensic video analysis.
                        </p>
                        <p>
                            Building upon the original hybrid CNN-RNN architecture, I implemented advanced interpretability mechanisms to visualize the model’s decision-making process:
                        </p>
                        <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                            <li><span className="text-blue-400 font-semibold">Grad-CAM:</span> Integrated to generate visual explanations, highlighting regions that triggered classification.</li>
                            <li><span className="text-purple-400 font-semibold">Heatmaps:</span> Novel visualization to track pixel-level anomalies across temporal sequences.</li>
                            <li><span className="text-white font-semibold">Enhanced UI:</span> Intuitive presentation of complex visual metrics to bridge the gap between inference and understanding.</li>
                        </ul>
                        <p>
                            This work not only refines the technical prowess of the detection engine but also addresses ethical AI concerns by providing verifiable proofs for every classification.
                        </p>
                        <hr className="border-gray-700 my-6" />
                        <p className="text-sm italic text-gray-400">
                            In collaboration with the <span className="text-gray-200 font-medium NOT-italic">University of Oviedo</span> and professor <span className="text-gray-200 font-medium NOT-italic">Cristian González García</span>.
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
