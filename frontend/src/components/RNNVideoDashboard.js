// CNNVideoDashboard.js
import React from 'react';
import {
  Box,
  Flex,
  Heading,
  Text,
  Image
} from '@chakra-ui/react';
import { ArrowForwardIcon } from '@chakra-ui/icons';
import { ResponsiveLine } from '@nivo/line';

const CNNVideoDashboard = ({ setVideoUploaded,setData,setLoading, data, setSelectedIndex, selectedIndex }) => (
  <Box flexDirection={'column'} h={'80%'}>
    <Flex direction='row' width={'100%'} alignContent={'flex-start'} >
      <Flex direction={'column'} alignContent={'flex-start'}>
        <Flex
          direction={'column'} 
          alignItems={'flex-start'}
          borderWidth='0.2em'
          borderRadius='0.5em'
          backgroundColor={'lightgrey'}
          padding={'1em'}
          marginLeft={'0.5em'}
          width={'25em'}>
          <Text><b>Nombre del video</b>: {data.predictions.id}</Text>
          <Text><b>Número de secuencias</b>: {data.predictions.data.length} </Text>
          <Text><b>Tamaño de las secuencias</b>: {data.sequenceLength} </Text>
          <Text><b>Mínimo registrado</b>: {(data.min*100).toFixed(2)}% </Text>
          <Text><b>Máximo registrado</b>: {(data.max*100).toFixed(2)}% </Text>
          <Text><b>Rango de valores</b>: {(data.range*100).toFixed(2)}% </Text>
          <Text><b>Varianza de los valores</b>: <b style={{ fontSize: '1.5em' ,color:'red'  }} >{(data.var*100).toFixed(2)}%</b> </Text>
          <Text><b>Media de los valores</b>: <b style={{ fontSize: '1.5em' ,color:'red'  }} >{(data.mean*100).toFixed(2)}%</b></Text>
        </Flex>
      </Flex>
      <Flex 
        direction={'column'}
        alignItems={'flex-start'}
        w='60em'
        padding={'0.7em'}
        borderWidth='0.2em'
        borderColor={'black'} 
        backgroundColor={'lightgrey'}
        margin={'0.5em'}
        borderRadius={'0.5em'}>
        <Heading as="h2" size="4xl" mb={15}>
          Frame seleccionado
        </Heading>
        <Flex 
          direction='row'
          w={'100%'}
          alignItems={'flex-start'}
          alignContent={'center'}
          marginLeft={'1em'}>
          <Image src={data.videoFrames[selectedIndex]} maxH={'20em'} maxW={'25em'} padding={'0.2em'} />
          <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'}/>
          <Image src={data.processedFrames[selectedIndex]} maxH='9.5em' maxW={'9.5em'} padding={'0.2em'} alignSelf={'center'} />
          <Flex direction={'column'} alignItems={'start'} marginLeft={'5em'}>
            <Text fontSize="lg">
              <b>Frame number</b>: {selectedIndex}
            </Text>
            <Text fontSize="lg">
              <b>Total frame count</b>: {data.nFrames}
            </Text>
            <Text fontSize="lg">
              <b>Fake %</b>: {(data.predictions.data[selectedIndex].y * 100).toFixed(2)}%
            </Text>
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
      backgroundColor={'lightgray'}
      borderRadius={'0.5em'}
      margin={'0.2em'}>
      <Heading as='h2' size='3xl' mb={13}>
        Análisis por frames
      </Heading>
      <div style={{ height: '18em', width: '100%'}} marginLeft='1em' marginRight='1em'>
        <ResponsiveLine
          data={[data.predictions]}
          margin={{ top: 20, right: 50, bottom: 70, left: 50 }}
          xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
          yScale={{ type: 'linear', min: 0, max: 1 , stacked: true }}
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
          pointColor={{ theme: 'background' }}
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
  </Box>
);

export default CNNVideoDashboard;
