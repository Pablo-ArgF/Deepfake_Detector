import React, { useState, useEffect } from 'react';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';
import { HiAcademicCap } from 'react-icons/hi';
import { useTranslation } from 'react-i18next';
import Joyride, { STATUS } from 'react-joyride';
import { trackFrameSelection, trackVideoPlayback } from '../utils/analytics';

const RNNVideoDashboard = ({ data, selectedIndex, setSelectedIndex, loading, runTutorial, setRunTutorial }) => {
    const { t } = useTranslation();

    const steps = [
        {
            target: '.main-dashboard-header',
            title: t('tutorial.rnn.step1_title'),
            content: t('tutorial.rnn.step1_content'),
            disableBeacon: true,
        },
        {
            target: '.stats-panel-rnn',
            title: t('tutorial.rnn.step2_title'),
            content: t('tutorial.rnn.step2_content'),
        },
        {
            target: '.video-player-rnn',
            title: t('tutorial.rnn.step3_title'),
            content: t('tutorial.rnn.step3_content'),
        },
        {
            target: '.chart-section-rnn',
            title: t('tutorial.rnn.step4_title'),
            content: t('tutorial.rnn.step4_content'),
        }
    ];

    const handleJoyrideCallback = (data) => {
        const { status } = data;
        if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
            setRunTutorial(false);
        }
    };

    const [threshold, setThreshold] = useState(0.5);
    const [aboveThreshold, setAboveThreshold] = useState(null);
    const [videoRef, setVideoRef] = useState(null);
    const [pieChartData, setPieChartData] = useState([
        { "id": "Above", "label": "Above", "value": 0 },
        { "id": "Below", "label": "Below", "value": 1 }
    ]);

    useEffect(() => {
        const predictionsData = data?.predictions?.data || [];
        const totalSequences = data?.nSequences || predictionsData.length || 0;

        // If predictionsData is frame-based (one point per frame), 
        // we count each unique sequence only once.
        let countAbove = 0;
        if (data?.nSequences && predictionsData.length > data.nSequences) {
            const step = Math.floor(predictionsData.length / data.nSequences);
            for (let i = 0; i < predictionsData.length; i += step) {
                if (predictionsData[i]?.y >= threshold) {
                    countAbove++;
                }
            }
        } else {
            countAbove = predictionsData.filter(prediction => prediction.y >= threshold).length;
        }

        const totalToDisplay = data?.nSequences || predictionsData.length || 1;

        setAboveThreshold(countAbove);
        setPieChartData([
            { "id": "Above", "label": t('common.deepfake'), "value": countAbove },
            { "id": "Below", "label": t('common.real'), "value": Math.max(0, totalToDisplay - countAbove) }
        ]);
    }, [threshold, data, t]);

    const handleThresholdChange = (event) => {
        let val = parseFloat(event.target.value) / 100;
        if (isNaN(val)) val = 0;
        if (val > 1) val = 1;
        setThreshold(val);
    };

    const handleVideoTimeUpdate = (e) => {
        const video = e.target;
        const duration = video.duration;
        const currentTime = video.currentTime;
        if (duration > 0) {
            const frameCount = data?.predictions?.data?.length || 1;
            const currentFrame = Math.floor((currentTime / duration) * (frameCount - 1));
            if (currentFrame !== selectedIndex) {
                setSelectedIndex(currentFrame);
            }
        }
    };

    const handleChartClick = (point) => {
        const index = point.data.x;
        setSelectedIndex(index);
        trackFrameSelection(index);
        if (videoRef) {
            const duration = videoRef.duration;
            const frameCount = data?.predictions?.data?.length || 1;
            if (isFinite(duration) && duration > 0 && frameCount > 1) {
                const targetTime = (index / (frameCount - 1)) * duration;
                if (isFinite(targetTime)) {
                    videoRef.currentTime = targetTime;
                }
            }
        }
    };

    useEffect(() => {
        if (data?.predictions?.data && (selectedIndex < 0 || selectedIndex >= (data?.predictions?.data?.length || 0))) {
            setSelectedIndex(0);
        }
        if (videoRef && data?.predictions?.data) {
            const duration = videoRef.duration;
            const frameCount = data?.predictions?.data?.length || 1;
            if (isFinite(duration) && duration > 0 && frameCount > 1) {
                const targetTime = (selectedIndex / (frameCount - 1)) * duration;
                if (isFinite(targetTime) && Math.abs(videoRef.currentTime - targetTime) > 0.3) {
                    videoRef.currentTime = targetTime;
                }
            }
        }
    }, [selectedIndex, data, videoRef]);

    if (loading || !data) {
        return (
            <div className="flex flex-col items-center justify-center p-20 min-h-[400px]">
                <div className="relative">
                    <div className="w-20 h-20 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin"></div>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-10 h-10 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin-slow"></div>
                    </div>
                </div>
                <div className="mt-8 text-center">
                    <p className="text-xl text-purple-400 font-black uppercase tracking-widest animate-pulse">{t('common.running_analysis')}</p>
                    <p className="mt-2 text-gray-500 font-medium">{t('common.processing_video')}</p>
                </div>
            </div>
        );
    }

    const stats = {
        total: data?.nSequences || 0,
        min: data?.min || 0,
        max: data?.max || 0,
        avg: data?.mean || 0,
        var: data?.var || 0,
        unique: new Set(data?.predictions?.data?.map(p => p.y.toFixed(2))).size
    };

    return (
        <div className="flex flex-col w-full gap-6">
            <Joyride
                steps={steps}
                run={runTutorial}
                continuous={true}
                showProgress={true}
                showSkipButton={true}
                callback={handleJoyrideCallback}
                styles={{
                    options: {
                        primaryColor: '#8b5cf6',
                        backgroundColor: '#1f2937',
                        textColor: '#fff',
                        arrowColor: '#1f2937',
                    }
                }}
                locale={{
                    back: t('tutorial.back'),
                    close: t('tutorial.last'),
                    last: t('tutorial.last'),
                    next: t('tutorial.next'),
                    skip: t('tutorial.skip'),
                }}
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
                {/* Global Stats Panel */}
                <div className="lg:col-span-5 flex flex-col gap-4 p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl h-full stats-panel-rnn">
                    <div className="bg-purple-600/20 border border-purple-500/50 p-4 rounded-2xl">
                        <p className="text-purple-300 text-sm font-bold uppercase tracking-wider mb-1">{t('common.video_name')}</p>
                        <h2 className="text-xl md:text-2xl font-black text-white truncate">{data?.predictions?.id}</h2>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        <StatCard label={t('common.frames')} value={stats.total} />
                        <StatCard label={t('common.min')} value={`${(stats.min * 100).toFixed(2)}%`} />
                        <StatCard label={t('common.max')} value={`${(stats.max * 100).toFixed(2)}%`} />
                        <StatCard label={t('common.unique')} value={stats.unique} />
                        <StatCard label={t('common.average')} value={`${(stats.avg * 100).toFixed(2)}%`} />
                        <StatCard label={t('common.variance')} value={`${(stats.var * 100).toFixed(2)}%`} />
                    </div>

                    <div className="mt-auto p-4 bg-gray-900/50 rounded-2xl border border-gray-700">
                        <div className="flex flex-col sm:flex-row items-center gap-4">
                            <div className="h-40 w-full sm:w-1/2">
                                <ResponsivePie
                                    data={pieChartData}
                                    margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
                                    innerRadius={0.6}
                                    padAngle={2}
                                    cornerRadius={4}
                                    colors={['#a855f7', '#1f2937']}
                                    enableArcLinkLabels={false}
                                    arcLabelsTextColor="#ffffff"
                                    theme={{ tooltip: { container: { background: '#1f2937', color: '#fff' } } }}
                                />
                            </div>
                            <div className="w-full sm:w-1/2 flex flex-col gap-3">
                                <label className="text-xs font-semibold text-gray-400">{t('common.decision_threshold')}</label>
                                <input
                                    type="number"
                                    max={100}
                                    min={0}
                                    value={threshold * 100}
                                    onChange={handleThresholdChange}
                                    className="bg-gray-800 border border-gray-600 rounded-xl px-4 py-2 focus:ring-2 focus:ring-purple-500 outline-none transition-all text-white font-bold"
                                />
                                {aboveThreshold !== null && (
                                    <p className="text-[10px] text-gray-500 italic">{t('common.sequences_above')}: <span className="text-purple-400 font-bold">{aboveThreshold}</span></p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Video and Prediction Integrated Card */}
                <div className="lg:col-span-7 flex flex-col p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl overflow-hidden focus-within:border-blue-500/50 transition-colors h-full video-player-rnn">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-bold">{t('common.frame')}: <span className="text-blue-400">{selectedIndex}</span></h2>
                        <div className="flex items-center gap-3">
                            <div className="bg-red-500/20 border border-red-500/50 px-4 py-1.5 rounded-full text-red-300 font-black text-xs uppercase tracking-wider">
                                {t('common.frame_prediction')}: {(data?.predictions?.data?.[selectedIndex]?.y * 100 || 0).toFixed(1)}%
                            </div>
                        </div>
                    </div>

                    <div className="flex-grow flex items-center justify-center min-h-0">
                        {/* Video Player with constrained height and proportional width */}
                        <div className="relative h-full max-h-[450px] aspect-video bg-black rounded-2xl overflow-hidden border-2 border-blue-500/30 shadow-2xl mx-auto">
                            <video
                                ref={(ref) => setVideoRef(ref)}
                                src={data?.isDemo ? '/demo/demo.mp4' : `/api/video/${data?.uuid}`}
                                className="w-full h-full object-contain"
                                controls
                                onTimeUpdate={handleVideoTimeUpdate}
                                onPlay={() => trackVideoPlayback('rnn')}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart Section */}
            <div className="flex flex-col gap-6 p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl chart-section-rnn">
                <div className="flex justify-between items-center">
                    <h2 className="text-2xl font-black dashboard-header-rnn">{t('common.sequence_block_analysis')}</h2>
                    <div className="flex items-center gap-2 text-[10px] font-bold text-gray-500 uppercase">
                        <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
                        <span>{t('common.prediction_per_20')}</span>
                    </div>
                </div>
                <div className="h-[20em] w-full bg-gray-900/50 rounded-2xl p-2 border border-gray-700">
                    {data?.predictions?.data ? (
                        <ResponsiveLine
                            data={[{ id: "Fake", data: data.predictions.data }]}
                            margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                            xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                            yScale={{ type: 'linear', min: 0, max: 1 }}
                            axisBottom={{ legend: t('common.frame_number'), legendOffset: 45, legendPosition: 'middle' }}
                            axisLeft={{ legend: t('common.fake_probability'), legendOffset: -50, legendPosition: 'middle', format: (v) => `${(v * 100).toFixed(0)}%` }}
                            theme={{
                                axis: {
                                    legend: { text: { fill: '#94a3b8', fontWeight: 600 } },
                                    ticks: { text: { fill: '#64748b' } }
                                },
                                grid: { line: { stroke: '#334155', strokeWidth: 1 } },
                                tooltip: { container: { background: '#1f2937', color: '#fff' } }
                            }}
                            colors={['#ef4444']}
                            lineWidth={3}
                            enableArea={true}
                            areaOpacity={0.1}
                            enablePoints={false}
                            useMesh={true}
                            markers={[
                                {
                                    axis: 'x',
                                    value: selectedIndex,
                                    lineStyle: { stroke: '#ffffff', strokeWidth: 2, strokeDasharray: '4 4' },
                                    legend: t('common.current'),
                                    legendOrientation: 'vertical',
                                    textStyle: { fill: '#ffffff', fontSize: 10, fontWeight: 'bold' }
                                },
                                {
                                    axis: 'y',
                                    value: threshold,
                                    lineStyle: { stroke: '#a855f7', strokeWidth: 2, strokeDasharray: '4 4' },
                                    legend: t('common.threshold'),
                                    legendPosition: 'top-left',
                                    textStyle: { fill: '#a855f7', fontSize: 10, fontWeight: 'bold' }
                                }
                            ]}
                            onClick={handleChartClick}
                        />
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-500 italic">{t('common.no_data')}</div>
                    )}
                </div>
            </div>
        </div>
    );
};

const StatCard = ({ label, value }) => (
    <div className="bg-gray-900 border border-gray-700/50 p-3 rounded-xl flex flex-col items-center">
        <span className="text-[10px] uppercase font-bold text-gray-500 mb-1">{label}</span>
        <span className="text-lg font-black text-white text-center">{value}</span>
    </div>
);

export default RNNVideoDashboard;
