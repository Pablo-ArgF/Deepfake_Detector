import React, { useState, useEffect, useRef } from 'react';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';
import { HiOutlineInformationCircle, HiXCircle, HiVideoCamera, HiPhotograph } from 'react-icons/hi';
import { IoMdTrendingUp } from 'react-icons/io';

const RNNVideoDashboard = ({ setVideoUploaded, setData, setLoading, loading, data, setSelectedIndex, selectedIndex }) => {
    const [threshold, setThreshold] = useState(0.5);
    const [aboveThreshold, setAboveThreshold] = useState(null);
    const [pieChartData, setPieChartData] = useState([{
        "id": "Above",
        "label": "Above",
        "value": 0
    },
    {
        "id": "Below",
        "label": "Below",
        "value": 1
    }]);

    useEffect(() => {
        const predictionsData = data?.predictions?.data || [];
        const countAbove = predictionsData.filter(prediction => prediction.y.toFixed(2) >= threshold).length;
        const normalizedAbove = countAbove / (data?.sequenceSize || 1);

        setAboveThreshold(normalizedAbove);
        setPieChartData([{
            "id": "Above",
            "label": "Above",
            "value": normalizedAbove
        },
        {
            "id": "Below",
            "label": "Below",
            "value": (data?.nSequences || 1) - normalizedAbove
        }]);
    }, [threshold, data]);

    const [isModalOpen, setIsModalOpen] = useState(false);
    const [imageUrl, setImageUrl] = useState('');
    const [viewMode, setViewMode] = useState('images'); // 'images' or 'video'
    const [videoRef, setVideoRef] = useState(null);

    const handleImageClick = (url) => {
        setImageUrl(url);
        setIsModalOpen(true);
    };

    const handleThresholdChange = (event) => {
        var thresholdValue = parseFloat(event.target.value).toFixed(2) / 100;
        if (thresholdValue > 1) thresholdValue = 1;
        if (event.target.value === '') thresholdValue = 0;
        setThreshold(thresholdValue);
    };

    const handleVideoTimeUpdate = (e) => {
        if (viewMode !== 'video') return;
        const video = e.target;
        const duration = video.duration;
        const currentTime = video.currentTime;
        if (duration > 0) {
            const frameCount = data?.predictions.data.length || 1;
            const currentFrame = Math.floor((currentTime / duration) * (frameCount - 1));
            if (currentFrame !== selectedIndex) {
                setSelectedIndex(currentFrame);
            }
        }
    };

    const handleChartClick = (point) => {
        const index = point.data.x;
        setSelectedIndex(index);
        if (viewMode === 'video' && videoRef) {
            const duration = videoRef.duration;
            const frameCount = data?.predictions.data.length || 1;
            videoRef.currentTime = (index / (frameCount - 1)) * duration;
        }
    };

    useEffect(() => {
        if (data?.predictions?.data && (selectedIndex < 0 || selectedIndex >= data.predictions.data.length)) {
            setSelectedIndex(0);
        }
        // Sync video player with selectedIndex if in video mode
        if (viewMode === 'video' && videoRef && data?.predictions?.data) {
            const frameRate = data.frameRate || 30;
            videoRef.currentTime = selectedIndex / frameRate;
        }
    }, [selectedIndex, data, setSelectedIndex, viewMode, videoRef]);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center p-12">
                <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                <p className="mt-4 text-blue-400 font-medium animate-pulse">Running sequence analysis...</p>
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
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

                {/* Stats Panel */}
                <div className="lg:col-span-5 flex flex-col gap-4 p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl">
                    <div className="bg-purple-600/20 border border-purple-500/50 p-4 rounded-2xl">
                        <p className="text-purple-300 text-sm font-bold uppercase tracking-wider mb-1">Video Name</p>
                        <h2 className="text-xl md:text-2xl font-black text-white truncate">{data?.predictions?.id}</h2>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        <StatCard label="Sequences" value={stats.total} />
                        <StatCard label="Min" value={`${(stats.min * 100).toFixed(2)}%`} />
                        <StatCard label="Max" value={`${(stats.max * 100).toFixed(2)}%`} />
                        <StatCard label="Unique" value={stats.unique} />
                        <StatCard label="Average" value={`${(stats.avg * 100).toFixed(2)}%`} />
                        <StatCard label="Variance" value={`${(stats.var * 100).toFixed(2)}%`} />
                    </div>

                    {/* Threshold & Pie */}
                    <div className="mt-4 p-4 bg-gray-900/50 rounded-2xl border border-gray-700">
                        <div className="flex flex-col sm:flex-row items-center gap-4">
                            <div className="h-48 w-full sm:w-1/2">
                                <ResponsivePie
                                    data={pieChartData}
                                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                                    innerRadius={0.6}
                                    padAngle={2}
                                    cornerRadius={4}
                                    activeOuterRadiusOffset={8}
                                    colors={['#a855f7', '#1f2937']}
                                    enableArcLinkLabels={false}
                                    arcLabelsTextColor="#ffffff"
                                    arcLabelsSkipAngle={10}
                                    theme={{
                                        tooltip: { container: { background: '#1f2937', color: '#fff' } }
                                    }}
                                />
                            </div>
                            <div className="w-full sm:w-1/2 flex flex-col gap-3">
                                <label className="text-sm font-semibold text-gray-400">Decision Threshold (%)</label>
                                <input
                                    type="number"
                                    max={100}
                                    min={0}
                                    value={threshold * 100}
                                    onChange={handleThresholdChange}
                                    placeholder="Enter %"
                                    className="bg-gray-800 border border-gray-600 rounded-xl px-4 py-2 focus:ring-2 focus:ring-purple-500 outline-none transition-all text-white font-bold"
                                />
                                {aboveThreshold !== null && (
                                    <p className="text-xs text-gray-500 italic">Sequences above: <span className="text-purple-400 font-bold">{aboveThreshold.toFixed(2)}</span></p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Selected Frame Details */}
                <div className="lg:col-span-7 flex flex-col p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl overflow-hidden">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-2">
                        <h2 className="text-xl font-bold">Frame: <span className="text-blue-400">{selectedIndex}</span></h2>
                        <div className="flex bg-gray-900 rounded-xl p-1 border border-gray-700">
                            <button
                                onClick={() => setViewMode('images')}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${viewMode === 'images' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-gray-200'}`}
                            >
                                <HiPhotograph /> <span className="text-xs font-bold uppercase">Images</span>
                            </button>
                            <button
                                onClick={() => setViewMode('video')}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${viewMode === 'video' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-gray-200'}`}
                            >
                                <HiVideoCamera /> <span className="text-xs font-bold uppercase">Video</span>
                            </button>
                        </div>
                        <div className="bg-red-500/20 border border-red-500/50 px-4 py-1 rounded-full text-red-300 font-bold">
                            Prediction: {(data?.predictions.data[selectedIndex]?.y * 100).toFixed(2)}% Deepfake
                        </div>
                    </div>

                    <div className="flex flex-col gap-6">
                        {viewMode === 'video' ? (
                            <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden border-2 border-blue-500/30">
                                <video
                                    ref={(ref) => setVideoRef(ref)}
                                    src={data.isDemo ? '/demo/demo.mp4' : `/api/video/${data.uuid}`}
                                    className="w-full h-full object-contain"
                                    controls
                                    onTimeUpdate={handleVideoTimeUpdate}
                                />
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 gap-4 animate-in fade-in duration-500">
                                <div className="flex flex-col gap-2">
                                    <div className="aspect-square bg-black/50 rounded-xl border border-gray-700 overflow-hidden">
                                        <ImageLink
                                            label="Processed Face"
                                            src={data.isDemo ? data.processedFrames[selectedIndex] : `/api/images/${data.uuid}/processed_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                        />
                                    </div>
                                </div>
                                <div className="flex flex-col gap-2">
                                    <div className="aspect-square bg-black/50 rounded-xl border border-gray-700 overflow-hidden">
                                        <ImageLink
                                            label="Original Frame"
                                            src={data.isDemo ? data.videoFrames[selectedIndex] : `/api/images/${data.uuid}/nonProcessed_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}

                        {viewMode === 'images' && (
                            <div className="grid grid-cols-2 gap-4 animate-in fade-in duration-500">
                                <div className="flex flex-col gap-2">
                                    <div className="aspect-square bg-black/50 rounded-xl border border-gray-700 overflow-hidden">
                                        <ImageLink
                                            label="Heatmap Face"
                                            src={data.isDemo ? data.heatmaps_face[selectedIndex] : `/api/images/${data.uuid}/heatmap_face_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            message={selectedIndex === 0 ? "Heatmap can not be computed for first frame of the video" : null}
                                        />
                                    </div>
                                </div>
                                <div className="flex flex-col gap-2">
                                    <div className="aspect-square bg-black/50 rounded-xl border border-gray-700 overflow-hidden">
                                        <ImageLink
                                            label="Full Heatmap"
                                            src={data.isDemo ? data.heatmaps[selectedIndex] : `/api/images/${data.uuid}/heatmap_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            message={selectedIndex === 0 ? "Heatmap can not be computed for first frame of the video" : null}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Bottom Section: Chart */}
            <div className="flex flex-col gap-6 p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl">
                <h2 className="text-2xl font-black">Sequence Analysis</h2>

                <div className="h-[20em] w-full bg-gray-900/50 rounded-2xl p-2 border border-gray-700">
                    {data?.predictions?.data ? (
                        <ResponsiveLine
                            data={[{ id: "Fake", data: data.predictions.data }]}
                            margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                            xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                            yScale={{ type: 'linear', min: 0, max: 1 }}
                            axisBottom={{
                                legend: 'Frame Number',
                                legendOffset: 45,
                                legendPosition: 'middle',
                                tickSize: 5,
                                tickPadding: 10,
                            }}
                            axisLeft={{
                                legend: 'Fake Probability',
                                legendOffset: -50,
                                legendPosition: 'middle',
                                tickSize: 5,
                                tickPadding: 10,
                            }}
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
                            enablePoints={false}
                            enableArea={true}
                            areaOpacity={0.1}
                            useMesh={true}
                            markers={[
                                {
                                    axis: 'x',
                                    value: selectedIndex,
                                    lineStyle: { stroke: '#ffffff', strokeWidth: 2, strokeDasharray: '4 4' },
                                    legend: 'Current',
                                    legendOrientation: 'vertical',
                                    textStyle: { fill: '#ffffff', fontSize: 10, fontWeight: 'bold' }
                                },
                                {
                                    axis: 'y',
                                    value: threshold,
                                    lineStyle: { stroke: '#a855f7', strokeWidth: 2, strokeDasharray: '4 4' },
                                    legend: `Threshold: ${Math.round(threshold * 100)}%`,
                                    legendPosition: 'top-left',
                                    textStyle: { fill: '#a855f7', fontSize: 10, fontWeight: 'bold' }
                                }
                            ]}
                            onClick={handleChartClick}
                        />
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-500 italic">No prediction data available</div>
                    )}
                </div>
            </div>

            {/* Modal */}
            {isModalOpen && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md p-4 animate-in fade-in duration-300"
                    onClick={() => setIsModalOpen(false)}
                >
                    <button className="absolute top-6 right-6 text-white text-4xl hover:text-red-500 transition-colors">
                        <HiXCircle />
                    </button>
                    <img
                        src={imageUrl}
                        alt="Fullscreen frame"
                        className="max-h-full max-w-full rounded-lg shadow-2xl transform scale-100 hover:scale-105 transition-transform duration-500 cursor-zoom-out"
                    />
                </div>
            )}
        </div>
    );
};

const StatCard = ({ label, value }) => (
    <div className="bg-gray-900 border border-gray-700/50 p-3 rounded-xl flex flex-col items-center">
        <span className="text-[10px] uppercase font-bold text-gray-500 mb-1">{label}</span>
        <span className="text-lg font-black text-white text-center">{value}</span>
    </div>
);

const ImageLink = ({ label, src, onClick, tooltip, large, message }) => (
    <div className="flex flex-col gap-2">
        <div className="flex items-center gap-1">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-tighter">{label}</span>
            {tooltip && (
                <div className="group relative">
                    <HiOutlineInformationCircle className="text-gray-600 cursor-help" />
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-gray-900 border border-gray-700 rounded-lg text-[10px] text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-2xl">
                        {tooltip}
                    </div>
                </div>
            )}
        </div>
        <div
            className={`relative overflow-hidden rounded-xl border-2 border-transparent ${!message ? 'hover:border-blue-500 cursor-zoom-in' : ''} transition-all group bg-black/50 ${large ? 'h-48 md:h-72' : 'h-24 md:h-32'} flex items-center justify-center`}
            onClick={() => !message && onClick ? onClick(src) : null}
        >
            {message ? (
                <p className="text-[10px] text-gray-500 font-medium px-4 text-center italic">{message}</p>
            ) : (
                <>
                    <img
                        src={src}
                        alt={label}
                        className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-110"
                    />
                    <div className="absolute inset-0 bg-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                </>
            )}
        </div>
    </div>
);

export default RNNVideoDashboard;
