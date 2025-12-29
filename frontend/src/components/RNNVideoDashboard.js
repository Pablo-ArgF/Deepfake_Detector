import React, { useState, useEffect } from 'react';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';

const RNNVideoDashboard = ({ setVideoUploaded, setData, setLoading, loading, data, setSelectedIndex, selectedIndex }) => {
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

    // Currently displayed image 
    const [videoFrameSrc, setVideoFrameSrc] = useState(data?.videoFrames[selectedIndex]);
    const [processedFrameSrc, setProcessedFrameSrc] = useState(data?.processedFrames[selectedIndex]);
    const [heatmapFrameSrc, setHeatmapFrameSrc] = useState(data?.heatmaps[selectedIndex]);
    const [heatmapFaceFrameSrc, setHeatmapFaceFrameSrc] = useState(data?.heatmaps_face[selectedIndex]);

    const handleThresholdChange = (event) => {
        var thresholdValue = parseFloat(event.target.value).toFixed(2) / 100;
        if (thresholdValue > 1) thresholdValue = 1;
        if (event.target.value === '') thresholdValue = 0;

        const predictionsData = data?.predictions?.data || [];
        const countAbove = predictionsData.filter(prediction => prediction.y.toFixed(2) >= thresholdValue).length;
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
    };

    useEffect(() => {
        if (data?.predictions?.data && (selectedIndex < 0 || selectedIndex >= data.predictions.data.length)) {
            setSelectedIndex(0);
        }
    }, [selectedIndex, data, setSelectedIndex]);

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
                        <h2 className="text-xl font-bold">Frame: <span className="text-purple-400">{selectedIndex}</span></h2>
                        <div className="bg-red-500/20 border border-red-500/50 px-4 py-1 rounded-full text-red-300 font-bold">
                            Prediction: {(data?.predictions?.data[selectedIndex]?.y * 100).toFixed(2)}% Fake
                        </div>
                    </div>

                    <div className="flex flex-col gap-6">
                        {/* Processed row */}
                        <div className="grid grid-cols-2 gap-4">
                            <ImageDisplay label="Heatmap (Full)" src={heatmapFrameSrc} />
                            <ImageDisplay label="Heatmap (Face)" src={heatmapFaceFrameSrc} />
                        </div>
                        {/* Source row */}
                        <div className="grid grid-cols-2 gap-4">
                            <ImageDisplay label="Original Frame" src={videoFrameSrc} />
                            <ImageDisplay label="Cropped Face" src={processedFrameSrc} />
                        </div>
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
                            colors={['#a855f7']}
                            lineWidth={3}
                            enablePoints={true}
                            pointSize={8}
                            pointColor="#a855f7"
                            pointBorderWidth={2}
                            pointBorderColor="#1f2937"
                            enableArea={true}
                            areaOpacity={0.1}
                            useMesh={true}
                            onClick={(point) => {
                                const index = point.index;
                                setSelectedIndex(index);
                                setVideoFrameSrc(data.videoFrames[index]);
                                setProcessedFrameSrc(data.processedFrames[index]);
                                setHeatmapFrameSrc(data.heatmaps[index]);
                                setHeatmapFaceFrameSrc(data.heatmaps_face[index]);
                            }}
                        />
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-500 italic">No prediction data available</div>
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

const ImageDisplay = ({ label, src }) => (
    <div className="flex flex-col gap-2">
        <span className="text-xs font-bold text-gray-400 uppercase tracking-tighter">{label}</span>
        <div className="relative overflow-hidden rounded-xl border-2 border-transparent hover:border-purple-500 transition-all bg-black/50 h-32 md:h-48 flex items-center justify-center">
            <img
                src={src}
                alt={label}
                className="max-h-full max-w-full object-contain"
            />
        </div>
    </div>
);

export default RNNVideoDashboard;
