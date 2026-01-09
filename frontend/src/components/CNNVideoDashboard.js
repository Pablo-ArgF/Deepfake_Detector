import React, { useState, useEffect } from 'react';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';
import { HiOutlineInformationCircle, HiXCircle, HiVideoCamera, HiPhotograph } from 'react-icons/hi';

const CNNVideoDashboard = ({ setVideoUploaded, setData, setLoading, data, setSelectedIndex, selectedIndex }) => {

    const [discartedIndexes, setDiscartedIndexes] = useState([]);
    const [threshold, setThreshold] = useState(0.5);
    const [aboveThreshold, setAboveThreshold] = useState(data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x) && prediction.y.toFixed(2) >= threshold).length);
    const [pieChartData, setPieChartData] = useState([{
        "id": "Above",
        "label": "Above",
        "value": aboveThreshold
    },
    {
        "id": "Below",
        "label": "Below",
        "value": data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x)).length - aboveThreshold
    }]);

    const [lineChartData, setLineChartData] = useState([{
        "id": "Fake",
        "data": data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(prediction.x)).map((prediction, index) => ({
            "x": prediction.x,
            "y": prediction.y
        }))
    }]);

    const [isModalOpen, setIsModalOpen] = useState(false);
    const [imageUrl, setImageUrl] = useState()
    const [viewMode, setViewMode] = useState('images'); // 'images' or 'video'
    const [videoRef, setVideoRef] = useState(null);
    const [imageErrors, setImageErrors] = useState({});
    const [isGenerating, setIsGenerating] = useState(false);

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

    useEffect(() => {
        const countAbove = data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x) && prediction.y.toFixed(2) >= threshold).length;
        setAboveThreshold(countAbove);
        setPieChartData([{
            "id": "Above",
            "label": "Above",
            "value": countAbove
        },
        {
            "id": "Below",
            "label": "Below",
            "value": data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x)).length - countAbove
        }]);
        setLineChartData([{
            "id": "Fake",
            "data": data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(prediction.x)).map((prediction, index) => ({
                "x": prediction.x,
                "y": prediction.y
            }))
        }]);
    }, [threshold, discartedIndexes, data?.predictions.data]);

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
            const frameCount = data?.predictions?.data?.length || 1;
            if (isFinite(duration) && duration > 0 && frameCount > 1) {
                const targetTime = (index / (frameCount - 1)) * duration;
                if (isFinite(targetTime)) {
                    videoRef.currentTime = targetTime;
                }
            }
        }
    };

    // Polling function to check if background generation is finished
    useEffect(() => {
        if (!data?.uuid || data?.isDemo) return;

        const hasErrors = Object.keys(imageErrors).length > 0;

        // If we have errors, we are definitely generating
        if (hasErrors) {
            setIsGenerating(true);
        }

        // If we aren't generating and no errors, no need to poll
        if (!isGenerating && !hasErrors) return;

        const interval = setInterval(async () => {
            try {
                // Clear errors to trigger a retry in the browser
                setImageErrors({});

                // We just clear them, the handleImageError will set isGenerating(true) 
                // if they fail again. The separate effect below will handle 
                // turning off the message after a stable period.
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [data?.uuid, data?.isDemo, isGenerating, Object.keys(imageErrors).length > 0]);

    // Separate effect to handle the "turning off" of isGenerating with a delay to avoid flicker
    useEffect(() => {
        const hasErrors = Object.keys(imageErrors).length > 0;
        if (!hasErrors && isGenerating) {
            const timer = setTimeout(() => {
                // Double check if still no errors after 2 seconds
                if (Object.keys(imageErrors).length === 0) {
                    setIsGenerating(false);
                }
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [Object.keys(imageErrors).length, isGenerating]);

    const handleImageError = (id) => {
        if (!imageErrors[id]) {
            setImageErrors(prev => ({ ...prev, [id]: true }));
        }
    };


    const discartCurrentFrame = () => {
        let newDiscartedIndexes;
        if (discartedIndexes.includes(selectedIndex)) {
            newDiscartedIndexes = discartedIndexes.filter(index => index !== selectedIndex);
        } else {
            newDiscartedIndexes = [...discartedIndexes, selectedIndex];
        }
        setDiscartedIndexes(newDiscartedIndexes);
        if (!data.isDemo) {
            fetch(`/api/recalculate/heatmaps/${data.uuid}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ discard_indices: newDiscartedIndexes })
            })
                .then(response => response.json())
                .then(updatedData => {
                    setData(prevData => ({
                        ...prevData,
                        heatmaps: updatedData.heatmaps,
                        heatmaps_face: updatedData.heatmaps_face
                    }));
                })
                .catch(error => console.error('Error recalculating heatmaps:', error));
        }
    };

    const filteredData = data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index));
    const stats = {
        total: filteredData.length,
        min: filteredData.length > 0 ? Math.min(...filteredData.map(p => p.y)) : 0,
        max: filteredData.length > 0 ? Math.max(...filteredData.map(p => p.y)) : 0,
        avg: filteredData.length > 0 ? filteredData.reduce((a, b) => a + b.y, 0) / filteredData.length : 0,
        unique: new Set(filteredData.map(p => p.y.toFixed(2))).size
    };

    return (
        <div className="flex flex-col w-full gap-6">
            {/* Top Section: Stats and Current Frame */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">

                {/* Stats Panel */}
                <div className="lg:col-span-5 flex flex-col gap-4 p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl h-full">
                    <div className="bg-blue-600/20 border border-blue-500/50 p-4 rounded-2xl">
                        <p className="text-blue-300 text-sm font-bold uppercase tracking-wider mb-1">Video Name</p>
                        <h2 className="text-xl md:text-2xl font-black text-white truncate">{data?.predictions.id}</h2>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        <StatCard label="Frames" value={stats.total} />
                        <StatCard label="Min" value={`${(stats.min * 100).toFixed(2)}%`} />
                        <StatCard label="Max" value={`${(stats.max * 100).toFixed(2)}%`} />
                        <StatCard label="Unique" value={stats.unique} />
                        <StatCard label="Average" value={`${(stats.avg * 100).toFixed(2)}%`} />
                        <StatCard label="Threshold" value={`${(threshold * 100).toFixed(0)}%`} />
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
                                    colors={['#3b82f6', '#1f2937']}
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
                                    value={threshold * 100}
                                    max={100}
                                    min={0}
                                    onChange={handleThresholdChange}
                                    className="bg-gray-800 border border-gray-600 rounded-xl px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-white font-bold"
                                />
                                <p className="text-xs text-gray-500 italic">Frames above threshold: <span className="text-blue-400 font-bold">{aboveThreshold}</span></p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Selected Frame Details */}
                <div className="lg:col-span-7 flex flex-col p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl overflow-hidden h-full">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-4">
                            <h2 className="text-xl font-bold">Frame: <span className="text-blue-400">{selectedIndex}</span></h2>
                            <div className="hidden md:flex bg-gray-900 rounded-lg p-0.5 border border-gray-700">
                                <button
                                    onClick={() => setViewMode('images')}
                                    className={`flex items-center gap-2 px-3 py-1 rounded-md transition-all ${viewMode === 'images' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-gray-200'}`}
                                >
                                    <HiPhotograph className="text-sm" /> <span className="text-[10px] font-bold uppercase">Images</span>
                                </button>
                                <button
                                    onClick={() => setViewMode('video')}
                                    className={`flex items-center gap-2 px-3 py-1 rounded-md transition-all ${viewMode === 'video' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-gray-200'}`}
                                >
                                    <HiVideoCamera className="text-sm" /> <span className="text-[10px] font-bold uppercase">Video</span>
                                </button>
                            </div>
                        </div>
                        <div className="bg-red-500/20 border border-red-500/50 px-4 py-1.5 rounded-full text-red-300 font-black text-xs uppercase tracking-wider">
                            Frame Prediction: {(data?.predictions.data[selectedIndex]?.y * 100).toFixed(1)}%
                        </div>
                    </div>

                    <div className="flex-grow flex flex-col justify-center">
                        {!discartedIndexes.includes(selectedIndex) || selectedIndex === 0 ? (
                            <div className="flex flex-col gap-6">
                                {viewMode === 'video' ? (
                                    <div className="flex-grow flex items-center justify-center min-h-0">
                                        <div className="relative h-full max-h-[450px] aspect-video bg-black rounded-2xl overflow-hidden border-2 border-blue-500/30 shadow-2xl mx-auto">
                                            <video
                                                ref={(ref) => setVideoRef(ref)}
                                                src={data.isDemo ? '/demo/demo.mp4' : `/api/video/${data.uuid}`}
                                                className="w-full h-full object-contain"
                                                controls
                                                onTimeUpdate={handleVideoTimeUpdate}
                                            />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-3 gap-4 animate-in fade-in duration-500">
                                        <ImageLink
                                            label="Processed"
                                            src={data.isDemo ? `/demo/frames/demo/images/processed_frame_${selectedIndex}.jpg` : `/api/images/${data.uuid}/processed_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            tooltip="Frame used for prediction after rotation and crop."
                                            isError={imageErrors[`proc_${selectedIndex}`]}
                                            onError={() => handleImageError(`proc_${selectedIndex}`)}
                                        />
                                        <ImageLink
                                            label="Heatmap"
                                            src={data.isDemo ? `/demo/frames/demo/images/heatmap_face_frame_${selectedIndex}.jpg` : `/api/images/${data.uuid}/heatmap_face_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            tooltip="Temporal changes compared to previous frame."
                                            message={selectedIndex === 0 ? "Heatmap can not be computed for first frame of the video" : null}
                                            isError={imageErrors[`hmf_${selectedIndex}`]}
                                            onError={() => handleImageError(`hmf_${selectedIndex}`)}
                                        />
                                        <ImageLink
                                            label="Grad-CAM"
                                            src={data.isDemo ? `/demo/frames/demo/images/gradcam_frame_${selectedIndex}.jpg` : `/api/images/${data.uuid}/gradcam_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            tooltip="AI attention regions for the prediction."
                                            isError={imageErrors[`gc_${selectedIndex}`]}
                                            onError={() => handleImageError(`gc_${selectedIndex}`)}
                                        />
                                    </div>
                                )}

                                {viewMode === 'images' && (
                                    <div className="grid grid-cols-2 gap-4 animate-in fade-in duration-500">
                                        <ImageLink
                                            label="Source Frame"
                                            src={data.isDemo ? `/demo/frames/demo/images/nonProcessed_frame_${selectedIndex}.jpg` : `/api/images/${data.uuid}/nonProcessed_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            large
                                            isError={imageErrors[`src_${selectedIndex}`]}
                                            onError={() => handleImageError(`src_${selectedIndex}`)}
                                        />
                                        <ImageLink
                                            label="Full Heatmap"
                                            src={data.isDemo ? `/demo/frames/demo/images/heatmap_frame_${selectedIndex}.jpg` : `/api/images/${data.uuid}/heatmap_frame_${selectedIndex}.jpg`}
                                            onClick={handleImageClick}
                                            large
                                            message={selectedIndex === 0 ? "Heatmap can not be computed for first frame of the video" : null}
                                            isError={imageErrors[`hm_${selectedIndex}`]}
                                            onError={() => handleImageError(`hm_${selectedIndex}`)}
                                        />
                                    </div>
                                )}
                                {isGenerating && (
                                    <div className="mt-2 p-3 bg-blue-900/20 border border-blue-500/30 rounded-xl flex items-center justify-center gap-3">
                                        <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                                        <span className="text-xs font-bold text-blue-300 uppercase tracking-widest">Generating visualizations in background...</span>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center p-12 opacity-50 italic">
                                <p>This frame has been discarded from analysis.</p>
                                <button
                                    onClick={discartCurrentFrame}
                                    className="mt-4 px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-xl transition-colors"
                                >
                                    Recover Frame
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Bottom Section: Chart */}
            <div className="flex flex-col gap-6 p-6 bg-gray-800 rounded-3xl border border-gray-700 shadow-xl">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                        <h2 className="text-2xl font-black">Frame Analysis</h2>
                        <button
                            onClick={discartCurrentFrame}
                            className={`px-6 py-2 rounded-xl font-bold transition-all shadow-md transform hover:scale-105 ${discartedIndexes.includes(selectedIndex)
                                ? 'bg-red-600 hover:bg-red-700 text-white'
                                : 'bg-gray-900 border border-gray-600 hover:border-blue-500 text-gray-300'
                                }`}
                        >
                            {discartedIndexes.includes(selectedIndex) ? 'Recover Frame' : 'Discard Frame'}
                        </button>
                    </div>

                    {discartedIndexes.length > 0 && (
                        <div className="flex items-center gap-2 overflow-x-auto max-w-full pb-2">
                            <span className="text-gray-500 text-xs font-bold uppercase whitespace-nowrap">{discartedIndexes.length} Discarded:</span>
                            {discartedIndexes.sort((a, b) => a - b).map((index) => (
                                <button
                                    key={index}
                                    onClick={() => {
                                        setSelectedIndex(index);
                                    }}
                                    className="flex-shrink-0 w-8 h-8 rounded-lg bg-gray-900 border border-red-900 text-red-500 text-xs flex items-center justify-center hover:bg-red-900 transition-colors"
                                >
                                    {index}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="h-[20em] w-full bg-gray-900/50 rounded-2xl p-2 border border-gray-700">
                    <ResponsiveLine
                        data={lineChartData}
                        margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                        xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                        yScale={{ type: 'linear', min: 0, max: 1 }}
                        axisBottom={{
                            legend: 'Frame Number',
                            legendOffset: 45,
                            legendPosition: 'middle',
                            tickSize: 5,
                            tickPadding: 10,
                            tickRotation: 0,
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
                        enablePoints={true}
                        pointSize={8}
                        pointColor="#ef4444"
                        pointBorderWidth={2}
                        pointBorderColor="#1f2937"
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
        <span className="text-lg font-black text-white">{value}</span>
    </div>
);

const ImageLink = ({ label, src, onClick, tooltip, large, message, isError, onError }) => {
    const [isReady, setIsReady] = React.useState(false);

    // Reset ready state when src changes to show loader for the new image
    React.useEffect(() => {
        setIsReady(false);
    }, [src]);

    const showLoader = (isError || !isReady) && !message;

    return (
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
                className={`relative overflow-hidden rounded-xl border-2 border-transparent ${(!message && !showLoader) ? 'hover:border-blue-500 cursor-zoom-in' : ''} transition-all group bg-black/50 ${large ? 'h-48 md:h-72' : 'h-24 md:h-32'} flex items-center justify-center`}
                onClick={() => !message && !showLoader && onClick(src)}
            >
                {message ? (
                    <p className="text-[10px] text-gray-500 font-medium px-4 text-center italic">{message}</p>
                ) : (
                    <>
                        {showLoader && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-gray-800/50 backdrop-blur-sm z-10 animate-in fade-in duration-300">
                                <div className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                                <p className="text-[10px] text-gray-400 font-bold uppercase animate-pulse">Generating...</p>
                            </div>
                        )}
                        <img
                            src={src}
                            alt={label}
                            className={`w-full h-full object-contain transition-all duration-500 group-hover:scale-110 ${!isReady ? 'opacity-0' : 'opacity-100'}`}
                            onLoad={() => setIsReady(true)}
                            onError={(e) => {
                                setIsReady(false);
                                if (onError) onError(e);
                            }}
                        />
                        {!showLoader && <div className="absolute inset-0 bg-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity" />}
                    </>
                )}
            </div>
        </div>
    );
};

export default CNNVideoDashboard;
