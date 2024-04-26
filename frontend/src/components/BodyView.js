// BodyView.js
import React, { useState } from 'react';
import {
  Heading,
  Text,
  FormControl,
  Image,
  Input,
  Flex,
  InputGroup,
  Button,
  Box,
  IconButton,
  Spinner
} from '@chakra-ui/react';
import { IoMdVideocam } from 'react-icons/io';
import { ArrowForwardIcon,ArrowBackIcon } from '@chakra-ui/icons'
import { ResponsiveLine } from '@nivo/line'


const BodyView = () => {
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(1);
  const [videoUploaded, setVideoUploaded] = useState(false); // Estado para controlar la visibilidad de los resultados
  const [loading, setLoading] = useState(false);


  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    console.log(file.name);

    if (!file) {
      setError('Por favor, selecciona un archivo de video.');
      return;
    }
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: file,
      });
      const data = await response.json();
      console.log(data);
      setData(data);
      setVideoUploaded(true);
    } catch (error) {
      setError('Error prediciendo DeepFakes:' + error);
      setLoading(false)
      setVideoUploaded(false)
      setData(null)
    }
  }; 

  return (
    <Box>
      {videoUploaded ? (
        <Box flexDirection={'column'} h={'80%'}>
          <Flex direction='row' width={'100%'} alignContent={'flex-start'} >
            <Flex direction={'column'} alignContent={'flex-start'}>
              <Flex direction={'row'} alignItems={'center'}>
                <IconButton
                  colorScheme='grey'
                  borderRadius={'1em'}
                  width={'3em'}
                  height={'3em'}
                  marginLeft={'1em'}
                  marginRight={'1.5em'}
                  marginTop={'1em'}
                  icon={<ArrowBackIcon boxSize={'2.8em'}/>}
                  onClick={() => {
                    setVideoUploaded(false) 
                    setData(null)
                    setLoading(false)
                  }}
                />
                <Heading alignSelf={'flex-start'} as="h1" size="5xl" padding={'0.1em'}>
                  Resultados
                </Heading>
              </Flex>
              <Flex
                direction={'column'} 
                alignItems={'flex-start'}
                borderWidth='0.2em'
                borderRadius='0.5em'
                backgroundColor={'lightgrey'}
                padding={'1em'}
                marginLeft={'0.5em'}
                width={'25em'}>
                <Text fontSize="lg">
                  <b>Nombre del video</b>: {data.predictions.id}
                </Text>
                <Text fontSize="lg">
                  <b>Frames analizados</b>: {data.predictions.data.length}
                </Text>
                <Text fontSize="lg" alignContent={'center'}>
                  <b>Probabilidad de que el video sea fake: </b> <b style={{ fontSize: '2em' ,color:'red'  }} >{(data.mean*100).toFixed(2)}%</b>
                </Text>
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
              borderRadius={'0.5em'}
              >
              <Heading as="h2" size="4xl" mb={15}>
                Frame seleccionado
              </Heading>
              <Flex 
                direction='row'
                w={'100%'}
                alignItems={'flex-start'}
                alignContent={'center'}
                marginLeft={'1em'}
                >
                <Image src={`data:image/jpeg;base64,${data.videoFrames[selectedIndex]}`} maxH={'20em'} maxW={'25em'} padding={'0.2em'} />
                <ArrowForwardIcon boxSize={'3em'} alignSelf={'center'}/>
                <Image src={`data:image/jpeg;base64,${data.processedFrames[selectedIndex]}`} maxH='9.5em' maxW={'9.5em'} padding={'0.2em'} alignSelf={'center'} />
                <Flex direction={'column'} alignItems={'start'} marginLeft={'5em'}>
                  <Text fontSize="lg" >
                    <b>Frame number</b>: {selectedIndex}
                  </Text>
                  <Text fontSize="lg">
                    <b>Total frame count</b>: {data.nFrames}
                  </Text>
                  <Text fontSize="lg">
                    <b>Fake %</b>: {data.predictions.data[selectedIndex].y * 100}%
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
            {/* Contenedor para la gráfica */}
            <Heading as='h2' size='3xl' mb={13} >
              Análisis por frames
            </Heading>
            <div style={{ height: '18em', width: '100%'}} marginLeft='1em' marginRight='1em'>
              <ResponsiveLine
                data={[data.predictions]}
                margin={{ top: 20, right: 50, bottom: 70, left: 50 }}
                xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                yScale={{ type: 'linear', min: 0, max: 'auto', stacked: true }}
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
      )

        :

        (
          <Flex direction="column" align="center" padding="2em">
            <Heading as="h1" size="5xl" mb={15}>
              Detección de DeepFakes
            </Heading>
            <Text textAlign="center" width="50%">
              El incipiente uso de las inteligencias artificiales ha hecho que la suplantación de identidad a través de deepFakes (contenido sintético generado por algoritmos de inteligencia artificial que combinan y superponen imágenes y vídeos existentes para crear uno nuevo, a menudo reemplazando la apariencia de una persona con la de otra). La herramienta que estás a punto de probar intenta ayudar en la identificación de este tipo de material sintético. Prueba a subir un video a continuación para comprobar si contiene DeepFakes o no.
            </Text>
            <Text color="red" marginTop="1em">
              {error}
            </Text>
            <FormControl padding="0.5em">
              <InputGroup marginTop="1em" marginBottom="1em">
                <Input
                  type="file"
                  id="videoInput"
                  accept="video/mp4"
                  onChange={handleVideoUpload}
                  style={{ display: 'none' }} // Hide the default file input
                />
                <Button as="label" leftIcon={<IoMdVideocam />} htmlFor="videoInput" colorScheme="teal" border='2px' borderColor='black.500' size="md">
                  Sube un video
                </Button>
              </InputGroup>
            </FormControl>
            {
              loading ? (
                <Spinner size='10xl' boxSize={'3em'}  thickness='0.2em' colorScheme='blue'/>
              ) : null
            }
          </Flex>

        )}
    </Box>
  );
};

export default BodyView;
