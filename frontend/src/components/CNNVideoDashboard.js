import React, { useState, useEffect } from 'react';
import {
    Flex,
    Grid,
    Heading,
    Text,
    Button,
    Input, FormControl, FormLabel
} from '@chakra-ui/react';
import { ArrowForwardIcon } from '@chakra-ui/icons';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';

const CNNVideoDashboard = ({ setVideoUploaded, setData, setLoading, data, setSelectedIndex, selectedIndex }) => {
    
    // Currently displayed image 
    const [videoFrameSrc, setVideoFrameSrc] = useState(data?.videoFrames[selectedIndex]);
    const [processedFrameSrc, setProcessedFrameSrc] = useState(data?.processedFrames[selectedIndex]);
    const [heatmapFrameSrc, setHeatmapFrameSrc] = useState(data?.heatmaps[selectedIndex]);
    const [heatmapFaceFrameSrc, setHeatmapFaceFrameSrc] = useState(data?.heatmaps_face[selectedIndex]);
    const [discartedIndexes, setDiscartedIndexes] = useState([]);
    const [threshold, setThreshold] = useState(0);
    const [aboveThreshold, setAboveThreshold] = useState(data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x) && prediction.y.toFixed(2) >= threshold).length);
    const [pieChartData, setPieChartData] = useState([{
            "id": "Above the threshold",
            "label": "Above the threshold",
            "value": aboveThreshold
        },
        {
            "id": "Below the threshold",
            "label": "Below the threshold",
            "value": data?.predictions.data.filter(prediction =>!discartedIndexes.includes(prediction.x)).length - aboveThreshold
        }]);

    const [lineChartData,setLineChartData] = useState([{
        "id": "Fake",
        "data": data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(prediction.x)).map((prediction, index) => ({
            "x": prediction.x,
            "y": prediction.y
            }))
        }]);
    const handleThresholdChange = (event) => {
        var thresholdValue = parseFloat(event.target.value).toFixed(2) / 100;
        if (thresholdValue > 1) thresholdValue = 1;
        if (event.target.value === '') thresholdValue = 0;
        setThreshold(thresholdValue);
    };

    useEffect(() => {  
        setAboveThreshold(data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x) && prediction.y.toFixed(2) >= threshold).length);
        setPieChartData([{
            "id": "Above the threshold",
            "label": "Above the threshold",
            "value": aboveThreshold
        },
        {
            "id": "Below the threshold",
            "label": "Below the threshold",
            "value": data?.predictions.data.filter(prediction => !discartedIndexes.includes(prediction.x)).length - aboveThreshold
        }]);
        setLineChartData([{
            "id": "Fake",
            "data": data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(prediction.x)).map((prediction, index) => ({
                "x": prediction.x,
                "y": prediction.y
            }))
        }]);
    }, [threshold, discartedIndexes, aboveThreshold, data?.predictions.data]);


    const discartCurrentFrame = () => {
        let newDiscartedIndexes;
        
        // if already discarded 
        if (discartedIndexes.includes(selectedIndex)) {
            // Recuperar el frame descartado
            newDiscartedIndexes = discartedIndexes.filter(index => index !== selectedIndex);
        } else {
            // Descartar el frame actual
            newDiscartedIndexes = [...discartedIndexes, selectedIndex];
        }
    
        setDiscartedIndexes(newDiscartedIndexes);
    
        fetch(`http://localhost/api/recalculate/heatmaps/${data.uuid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ discard_indices: newDiscartedIndexes })
        })
        .then(response => response.json())
        .then(updatedData => {
            setData(prevData => ({
                ...prevData,
                heatmaps: updatedData.heatmaps,
                heatmaps_face: updatedData.heatmaps_face
            }));
            setHeatmapFrameSrc(updatedData.heatmaps[selectedIndex]);
            setHeatmapFaceFrameSrc(updatedData.heatmaps_face[selectedIndex]);
        })
        .catch(error => console.error('Error recalculating heatmaps:', error));
    };


    return (
        <Flex direction='column' width={'100%'} placeContent={'flex-start'} h={'80%'} w='100%'>
            <Flex direction='column' width={'100%'} alignContent={'flex-start'} >
                <Flex direction='row' width={'100%'} alignContent={'flex-start'} flex-wrap='wrap'>
                    <Flex
                        direction={'column'}
                        wrap={'wrap'}
                        gap={'0.5em'}
                        alignItems={'flex-start'}
                        borderWidth='0.2em'
                        borderRadius='0.5em'
                        backgroundColor={'#A7E6FF'}
                        padding={'1em'}
                        marginLeft={'0.5em'}
                        width={'35%'}>
                        <Flex direction={'column'} w='98%' padding='0.5em' alignItems='flex-start' backgroundColor='#3572EF' borderRadius={'0.25em'}>
                            <Text textColor={'black'} margin={'0.25em'}><b>Name of the video</b></Text>
                            <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{data?.predictions.id}</Text>
                        </Flex>
                        <Grid
                            w='98%'
                            templateRows='repeat(2, 6.5em)'
                            templateColumns='repeat(3, 33.333%)'
                            gap={'0.25em'}
                        >
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Number of analyzed frames</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).length}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Minimum</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(Math.min(...data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).map(prediction => prediction.y)) * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Maximum</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(Math.max(...data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).map(prediction => prediction.y)) * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Number of different values</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{new Set(data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).map(prediction => prediction.y.toFixed(2))).size}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Variance of the values</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).reduce((acc, prediction) => acc + Math.pow(prediction.y - data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).reduce((a, b) => a + b.y, 0) / data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).length, 2), 0) / data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).length * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Average of the values</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).reduce((a, b) => a + b.y, 0) / data?.predictions.data.filter((prediction, index) => !discartedIndexes.includes(index)).length * 100).toFixed(2)}%</Text>
                            </Flex>
                        </Grid>
                        {/* Threshold Control and Pie Chart */}
                        <Flex direction={'column'} padding={'0.5em'} width={'98%'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                            <Text textColor={'black'} margin={'0.25em'}><b>Proportion above threshold</b>:</Text>
                            <Flex direction='row' height='12em' width={'100%'}>
                                <div style={{ height: '17em', width: '75%', overflow: 'hidden' }} marginLeft='1em' marginRight='1em'>
                                    {aboveThreshold != null ?
                                        <ResponsivePie
                                            data={pieChartData}
                                            margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
                                            innerRadius={0.5}
                                            padAngle={0.7}
                                            cornerRadius={3}
                                            activeOuterRadiusOffset={8}
                                            colors={{ scheme: 'dark2' }}
                                            borderWidth={1}
                                            borderColor={{
                                                from: 'color',
                                                modifiers: [
                                                    [
                                                        'darker',
                                                        0.2
                                                    ]
                                                ]
                                            }}
                                            enableArcLinkLabels={false}
                                            arcLabelsSkipAngle={10}
                                            arcLabelsTextColor={{
                                                from: 'color',
                                                modifiers: [
                                                    [
                                                        'brighter',
                                                        3
                                                    ]
                                                ]
                                            }}
                                            animate={true}
                                            legends={[
                                                {
                                                    anchor: 'top-left',
                                                    direction: 'column',
                                                    justify: false,
                                                    translateX: -80,
                                                    translateY: -30,
                                                    itemsSpacing: 3,
                                                    itemWidth: 100,
                                                    itemHeight: 18,
                                                    itemTextColor: '#A7E6FF',
                                                    itemDirection: 'left-to-right',
                                                    itemOpacity: 1,
                                                    symbolSize: 18,
                                                    symbolShape: 'circle',
                                                    effects: [
                                                        {
                                                            on: 'hover',
                                                            style: {
                                                                itemTextColor: 'white'
                                                            }
                                                        }
                                                    ]
                                                }
                                            ]}
                                        /> :
                                        <Flex justifyContent={'center'}><Text textColor={'black'} margin={'0.25em'}><b>Introduce the decision <br />threshold</b></Text></Flex>
                                    }
                                </div>
                                <Flex direction="column" width="35%" h="100%" justifyContent="flex-end">
                                    <FormControl>
                                        <FormLabel htmlFor="threshold">Introduce the threshold (%)</FormLabel>
                                        <Input
                                            id="threshold"
                                            type="number"
                                            width="90%"
                                            height="1.8em"
                                            max={100}
                                            min={0}
                                            placeholder="Introduce the threshold (%)"
                                            onChange={handleThresholdChange}
                                        />
                                    </FormControl>
                                </Flex>
                            </Flex>
                        </Flex>
                    </Flex>
                    <Flex
                        direction={'column'}
                        alignItems={'flex-start'}
                        w='65%'
                        padding={'0.7em'}
                        borderWidth='0.2em'
                        borderColor={'black'}
                        backgroundColor={'#786EDF'}
                        marginLeft={'0.5em'}
                        marginRight={'0.5em'}
                        borderRadius={'0.5em'}>
                        <Heading as="h2" size="4xl" mb={15} textColor={'black'}>
                            Selected frame
                        </Heading>
                        <Flex
                            direction='row'
                            w={'100%'}
                            alignItems={'flex-start'}
                            alignContent={'center'}
                            marginLeft={'1em'}
                            gap={'0.5em'}>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Frame number</b></Text>
                                <Text textColor={'black'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{selectedIndex}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Prediction for the frame</b></Text>
                                <Text textColor={'black'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.predictions.data[selectedIndex]?.y * 100).toFixed(2)}% Fake</Text>
                            </Flex>
                        </Flex>
                        <Flex
                            direction='row'
                            w={'100%'}
                            alignItems={'flex-start'}
                            alignContent={'center'}
                            marginLeft={'1em'}
                            marginTop={'2em'}
                        >
                            {!discartedIndexes.includes(selectedIndex) || selectedIndex === 0 ?
                            <Flex
                            direction='row'
                            alignItems={'flex-start'}
                            alignContent={'center'}
                            >
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}>
                                    <b>Heatmap for the frame</b>
                                </Text>
                                <Flex direction={'row'}>
                                    <img
                                        src= {'http://localhost/api/images/'+data.uuid+'/heatmap_frame_'+selectedIndex+'.jpg'}
                                        alt='Mapa de calor para el fotograma seleccionado'
                                        style={{
                                            maxHeight: '20em',
                                            maxWidth: '25em',
                                            padding: '0.2em',
                                            marginRight: '0.5em'
                                        }}
                                    />
                                </Flex>
                            </Flex>
                            <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'} />
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}>
                                    <b>Heatmap for the processed frame</b>
                                </Text>
                                <Flex direction={'row'}>
                                    <img
                                        src= {'http://localhost/api/images/'+data.uuid+'/heatmap_face_frame_'+selectedIndex+'.jpg'}
                                        alt='Mapa de calor para el fotograma seleccionado tras el procesamiento'
                                        style={{
                                            maxHeight: '20em',
                                            maxWidth: '25em',
                                            padding: '0.2em'
                                        }}
                                        alignSelf={'center'}
                                    />
                                </Flex>
                            </Flex>
                            </Flex>
                            :<div></div>
                            }
                            {/* Video Frame */}
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'} marginLeft={'1em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}>
                                    <b>Extracted frame from the video</b>
                                </Text>
                                {/* Display video frame from URL */}
                                <img
                                    src= {'http://localhost/api/images/'+data.uuid+'/nonProcessed_frame_'+selectedIndex+'.jpg'}
                                    alt='Fotograma real del video'
                                    style={{
                                        maxHeight: '20em',
                                        maxWidth: '25em',
                                        padding: '0.2em'
                                    }}
                                />
                            </Flex>

                            <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'} />

                            {/* Processed Frame */}
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}>
                                    <b>Frame used for the prediction</b>
                                </Text>
                                {/* Display processed frame from URL */}
                                <img
                                    src= {'http://localhost/api/images/'+data.uuid+'/processed_frame_'+selectedIndex+'.jpg'}
                                    alt='Fotograma recortado utilizado para la detección'
                                    style={{
                                        maxHeight: '20em',
                                        maxWidth: '25em',
                                        padding: '0.2em',
                                        alignSelf: 'center'
                                    }}
                                />
                            </Flex>

                        </Flex>

                    </Flex>
                </Flex>
                {/* Frame Analysis Chart */}
                <Flex
                    height={'100%'}
                    padding='0.5em'
                    direction='column'
                    alignContent={'flex-start'}
                    alignItems={'start'}
                    backgroundColor={'white'}
                    borderWidth='0.3em'
                    borderColor={'black'}
                    borderRadius={'0.5em'}
                    borderStyle={'solid'}
                    margin={'0.5em'}
                >
                    <Flex direction='row' width={'100%'} alignContent={'flex-start'} alignItems={'center'}> 
                        <Heading as='h2' size='3xl' mb={13} textColor={'#00201D'}>
                            Frame analysis
                        </Heading>
                        <Button 
                            marginLeft={'1em'}
                            marginTop={'1em'}
                            borderRadius={'1em'}
                            backgroundColor={ discartedIndexes.includes(selectedIndex) ? 'red' : 'black'}
                            textColor={'white'}
                            fontSize={'1em'}
                            padding={'0.7em'}
                            colorScheme='blue'
                            w={'10em'}
                            h={'3em'}
                            onClick={() => discartCurrentFrame()}
                            cursor='pointer'
                        >
                            {discartedIndexes.includes(selectedIndex) ? 'Recover Frame' : 'Discard Frame'}
                        </Button>
                        {discartedIndexes.length > 0 ?
                            <Flex direction='row' alignContent={'flex-start'} alignItems={'center'}> 
                                <Text textColor={'#00201D'} fontSize={'1em'} fontFamily={'revert'} margin={'0.25em'}>
                                    {discartedIndexes.length} frames discarded
                                </Text> 
                                {discartedIndexes.sort().map((index) => (
                                    <Button 
                                        marginLeft={'1em'}
                                        marginTop={'1em'}
                                        borderRadius={'1em'}
                                        backgroundColor={'black'}
                                        textColor={'white'}
                                        fontSize={'1em'}
                                        padding={'0.7em'}
                                        colorScheme='blue'
                                        w={'4em'}
                                        h={'3em'}
                                        onClick={() => {
                                            setSelectedIndex(index); // Set the selected index
                                            setVideoFrameSrc(data?.videoFrames[index]);
                                            setProcessedFrameSrc(data?.processedFrames[index]);
                                            setHeatmapFrameSrc(data?.heatmaps[index]);
                                            setHeatmapFaceFrameSrc(data?.heatmaps_face[index]);
                                        }}
                                        cursor='pointer'
                                    >
                                        {index}
                                    </Button>
                                ))}
                            </Flex>   
                            : null
                        }
                    </Flex>
                    <div style={{ height: '18em', width: '100%' }} marginLeft='1em' marginRight='1em'>
                    <ResponsiveLine
                        data={lineChartData}
                        margin={{ top: 20, right: 50, bottom: 70, left: 50 }}
                        xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                        yScale={{ type: 'linear', min: 0, max: 1, stacked: true }}
                        yFormat=" >-.2f"
                        axisBottom={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'Número de frame',
                            legendOffset: 36,
                            legendPosition: 'middle',
                            truncateTickAt: 0
                        }}
                        axisLeft={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: '% Fake',
                            legendOffset: -40,
                            legendPosition: 'middle',
                            truncateTickAt: 0
                        }}
                        enableTouchCrosshair={true}
                        useMesh={true}
                        pointBorderWidth={2}
                        pointBorderColor={{ from: 'serieColor' }}
                        colors={{ scheme: 'set1' }}
                        curve="monotoneX"
                        onClick={(point) => {
                            const index = point.data.x;
                            setSelectedIndex(index); // Set the selected index
                            setVideoFrameSrc(data?.videoFrames[index]);
                            setProcessedFrameSrc(data?.processedFrames[index]);
                            setHeatmapFrameSrc(data?.heatmaps[index]);
                            setHeatmapFaceFrameSrc(data?.heatmaps_face[index]);
                        }}
                    />
                    </div>
                </Flex>
            </Flex>
        </Flex >
    );
};

export default CNNVideoDashboard;
