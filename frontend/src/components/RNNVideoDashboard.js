import React, { useState, useEffect } from 'react';
import {
    Flex,
    Grid,
    Heading,
    Text,
    Image,
    Input,
    Spinner,FormControl,FormLabel
} from '@chakra-ui/react';
import { ArrowForwardIcon } from '@chakra-ui/icons';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';

const RNNVideoDashboard = ({ setVideoUploaded, setData, setLoading,loading, data, setSelectedIndex, selectedIndex }) => {
    const [aboveThreshold, setAboveThreshold] = useState(null);
    const [pieChartData,setPieChartData] = useState([{
        "id": "Por encima del umbral",
        "label": "Por encima del umbral",
        "value": aboveThreshold
    },
    {
        "id": "Por debajo del umbral",  
        "label": "Por debajo del umbral",
        "value": data?.nSequences - aboveThreshold
    }]);

    // Currently displayed image 
    const [videoFrameSrc, setVideoFrameSrc] = useState(data?.videoFrames[selectedIndex]);
    const [processedFrameSrc, setProcessedFrameSrc] = useState(data?.processedFrames[selectedIndex]);
    const [heatmapFrameSrc, setHeatmapFrameSrc] = useState(data?.heatmaps[selectedIndex]);
    const [heatmapFaceFrameSrc, setHeatmapFaceFrameSrc] = useState(data?.heatmaps_face[selectedIndex]);

    const handleThresholdChange = (event) => {
        // threshold value is the value rounded to 2 decimal positions and divided by 100
        var thresholdValue = parseFloat(event.target.value).toFixed(2) / 100;
        if (thresholdValue > 1)
          thresholdValue = 1;
        if (event.target.value === '') {
          thresholdValue = 0;
        }
        
        const predictionsData = data?.predictions?.data || [];
        const aboveThresholdTmp = (predictionsData.filter(prediction => prediction.y.toFixed(2) >= thresholdValue).length) / data?.sequenceSize;
        setAboveThreshold(aboveThresholdTmp);
        setPieChartData([{
            "id": "Above the threshold",
            "label": "Above the threshold",
            "value": aboveThresholdTmp
        },
        {
            "id": "Below the threshold",
            "label": "Below the threshold",
            "value": data?.nSequences - aboveThresholdTmp
        }]);
    };

    useEffect(() => {
        if (selectedIndex < 0 || selectedIndex >= data?.predictions?.data?.length) {
            setSelectedIndex(1);
        }
    }, [selectedIndex, data, setSelectedIndex]);

    return (
      <div>
        {loading ? 
        (
        <Flex direction='column' width={'100%'} alignItems={'center'} >
            <Spinner size='10xl' boxSize={'3em'}  thickness='0.2em' colorScheme='blue'/>
        </Flex>
        )    
        :
        (
        <Flex direction='column' width={'100%'} placeContent={'flex-start'} h={'80%'} w='100%'>
            <Flex direction='column' width={'100%'} alignContent={'flex-start'} >
                <Flex direction='row' width={'100%'} alignContent={'flex-start'} flex-wrap='wrap' >
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
                            <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{data?.predictions?.id}</Text>
                        </Flex>
                        <Grid
                            w='98%'
                            templateRows='repeat(2, 6.5em)'
                            templateColumns='repeat(3, 33.333%)'
                            gap={'0.25em'}
                        >
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Number of analyzed</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{data?.nSequences}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Minimun</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.min * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Maximun</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.max * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Number of different values</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{new Set(data?.predictions?.data?.map(prediction => prediction.y.toFixed(2))).size}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Variance of the values</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.var * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Average of the values</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.mean * 100).toFixed(2)}%</Text>
                            </Flex>
                        </Grid>

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
                                        <Flex justifyContent={'center'}><Text textColor={'black'} margin={'0.25em'}><b>Introduce the decision <br/>threshold</b></Text></Flex>
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
                                    <Text textColor={'#170C8A'} margin={'0.25em'}><b>Prediction for the sequence's frame</b></Text>
                                    <Text textColor={'black'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.predictions?.data[selectedIndex]?.y * 100).toFixed(2)}% Fake</Text>
                                </Flex>
                        </Flex>
                        <Flex
                            direction='row'
                            w={'100%'}
                            alignItems={'flex-start'}
                            alignContent={'center'}
                            marginLeft={'1em'}
                            marginTop={'2em'}>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'} marginRight={'0.5em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Heatmap for the frame</b></Text>
                                <Image src={heatmapFrameSrc} alt='Heatmap for the frame' maxH={'20em'} maxW={'25em'} padding={'0.2em'} />
                            </Flex>
                            <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'} />
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'} marginRight={'0.5em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Heatmap for the processed frame</b></Text>
                                <Image src={heatmapFaceFrameSrc} alt='Heatmap for the processed frame' maxH={'20em'} maxW={'25em'} padding={'0.2em'} alignSelf={'center'} />
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Frame extracted from the video</b></Text>
                                <Image src={videoFrameSrc} alt='Frame extracted from the video' maxH={'20em'} maxW={'25em'} padding={'0.2em'} />
                            </Flex>
                            <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'} />
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Cropped frame used for detection</b></Text>
                                <Image src={processedFrameSrc} alt='Cropped frame used for detection' maxH='20em' maxW={'25em'} padding={'0.2em'} alignSelf={'center'} />
                            </Flex>
                        </Flex>
                    </Flex>
                </Flex>
                                            
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
                    <Heading as='h2' size='3xl' mb={13} textColor={'#00201D'}>
                        Sequence analysis
                    </Heading>
                    <div style={{ height: '18em', width: '100%' }} marginLeft='1em' marginRight='1em'>
                        {data?.predictions != null?
                        <ResponsiveLine
                            data={[data?.predictions] || []}
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
                                const index = point.index;
                                setSelectedIndex(index); // Set the selected index
                                setVideoFrameSrc(data.videoFrames[index]); 
                                setProcessedFrameSrc(data.processedFrames[index]); 
                                setHeatmapFrameSrc(data.heatmaps[index]); 
                                setHeatmapFaceFrameSrc(data.heatmaps_face[index]);  
                            }}/>
                            :
                            <div></div>}
                    </div>
                </Flex>
            </Flex>
        </Flex >
        )
      }</div>
    );
};

export default RNNVideoDashboard;