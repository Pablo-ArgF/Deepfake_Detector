import React, { useState } from 'react';
import {
    Flex,
    Grid,
    Heading,
    Text,
    Image,
    Input, FormControl, FormLabel
} from '@chakra-ui/react';
import { ArrowForwardIcon } from '@chakra-ui/icons';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';

const CNNVideoDashboard = ({ setVideoUploaded, setData, setLoading, data, setSelectedIndex, selectedIndex }) => {
    const [aboveThreshold, setAboveThreshold] = useState(null);
    const [pieChartData,setPieChartData] = useState([{
        "id": "Por encima del umbral",
        "label": "Por encima del umbral",
        "value": aboveThreshold
    },
    {
        "id": "Por debajo del umbral",  
        "label": "Por debajo del umbral",
        "value": data?.nFrames - aboveThreshold
    }]);

    const handleThresholdChange = (event) => {
        // threshold value is the value rounded to 2 decimal positions and divided by 100
        var thresholdValue = parseFloat(event.target.value).toFixed(2) / 100;
        if (thresholdValue > 1)
            thresholdValue = 1;
        if (event.target.value === '') {
            thresholdValue = 0;
        }
        const aboveThresholdTmp = data?.predictions.data.filter(prediction => prediction.y.toFixed(2)  >= thresholdValue).length;
        setAboveThreshold(aboveThresholdTmp);
        setPieChartData([{
            "id": "Por encima del umbral",
            "label": "Por encima del umbral",
            "value": aboveThresholdTmp
        },
        {
            "id": "Por debajo del umbral",
            "label": "Por debajo del umbral",
            "value": data?.nFrames - aboveThresholdTmp
        }]);
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
                            <Text textColor={'black'} margin={'0.25em'}><b>Nombre del video</b></Text>
                            <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{data?.predictions.id}</Text>
                        </Flex>
                        <Grid
                            w='98%'
                            templateRows='repeat(2, 6.5em)'
                            templateColumns='repeat(3, 33.333%)'
                            gap={'0.25em'}
                        >
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Frames analizados</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{data?.predictions.data.length}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Mínimo registrado</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.min * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Máximo registrado</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.max * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Número de valores distintos</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{new Set(data?.predictions.data.map(prediction => prediction.y.toFixed(2))).size}</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Varianza de los valores</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.var * 100).toFixed(2)}%</Text>
                            </Flex>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                                <Text textColor={'black'} margin={'0.25em'}><b>Media de los valores</b></Text>
                                <Text textColor={'white'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.mean * 100).toFixed(2)}%</Text>
                            </Flex>
                        </Grid>

                        <Flex direction={'column'} padding={'0.5em'} width={'98%'} backgroundColor='#3572EF' borderRadius={'0.25em'}>
                            <Text textColor={'black'} margin={'0.25em'}><b>Proporción por encima del umbral</b>:</Text>
                            <Flex direction='row' height='12em' width={'100%'}>                                
                                <div style={{ height: '17em', width:'75%', overflow:'hidden' }} marginLeft='1em' marginRight='1em'>
                                    { aboveThreshold != null?
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
                                    />
                                    :
                                    <Flex justifyContent={'center'}><Text textColor={'black'} margin={'0.25em'}><b>Introduzca un umbral <br/>de decisión</b></Text></Flex>
                                    }
                                </div>
                                <Flex direction="column" width="35%" h="100%" justifyContent="flex-end">
                                  <FormControl>
                                    <FormLabel htmlFor="threshold">Introduce el umbral (%)</FormLabel>
                                    <Input
                                      id="threshold"
                                      type="number"
                                      width="90%"
                                      height="1.8em"
                                      max={100}
                                      min={0}
                                      placeholder="Introduce el umbral (%)"
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
                            Frame seleccionado
                        </Heading>
                        <Flex
                            direction='row'
                            w={'100%'}
                            alignItems={'flex-start'}
                            alignContent={'center'}
                            marginLeft={'1em'}
                            gap={'0.5em'}>
                                <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                    <Text textColor={'#170C8A'} margin={'0.25em'}><b>Número del frame</b></Text>
                                    <Text textColor={'black'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{selectedIndex}</Text>
                                </Flex>
                                <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                    <Text textColor={'#170C8A'} margin={'0.25em'}><b>Predicción para el frame</b></Text>
                                    <Text textColor={'black'} fontSize={'1.6em'} fontFamily={'revert'} margin={'0.25em'}>{(data?.predictions.data[selectedIndex]?.y * 100).toFixed(2)}% Fake</Text>
                                </Flex>
                        </Flex>
                        <Flex
                            direction='row'
                            w={'100%'}
                            alignItems={'flex-start'}
                            alignContent={'center'}
                            marginLeft={'1em'}
                            marginTop={'2em'}>
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Frame extraido del video</b></Text>
                                <Image src={data?.videoFrames[selectedIndex]} alt='Fotograma real del video' maxH={'20em'} maxW={'25em'} padding={'0.2em'} />
                            </Flex>
                            <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'} />
                            <Flex direction={'column'} padding={'0.5em'} backgroundColor='#AEAAEE' borderRadius={'0.25em'}>
                                <Text textColor={'#170C8A'} margin={'0.25em'}><b>Frame utilizado para la predicción</b></Text>
                                <Image src={data?.processedFrames[selectedIndex]} alt='Fotograma recortado utilizado para la detección' maxH='9.5em' maxW={'9.5em'} padding={'0.2em'} alignSelf={'center'} />
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
                        Análisis por frames
                    </Heading>
                    <div style={{ height: '18em', width: '100%' }} marginLeft='1em' marginRight='1em'>
                        <ResponsiveLine
                            data={[data?.predictions]}
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
                            onClick={(data) => {
                                setSelectedIndex(data.index);
                            }}
                        />
                    </div>
                </Flex>
            </Flex>
        </Flex >
    );
};

export default CNNVideoDashboard;