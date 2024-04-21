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
  Center
} from '@chakra-ui/react';
import { IoMdVideocam } from 'react-icons/io';
import { ResponsiveLine } from '@nivo/line'


// Import the results.json file
import resultsData from './results.json';


const BodyView = () => {
  const [error, setError] = useState('');
  const [data, setData] = useState(resultsData);
  const [selectedIndex, setSelectedIndex] = useState(1);
  const [videoUploaded, setVideoUploaded] = useState(true); // Estado para controlar la visibilidad de los resultados


  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    console.log(file.name);

    if (!file) {
      setError('Por favor, selecciona un archivo de video.');
      return;
    }
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: file,
      });
      const data = await response.json();
      console.log(data);
      setData(data);
      setVideoUploaded(true);
    } catch (error) {
      console.error('Error prediciendo DeepFakes:', error);
    }
  };

  return (
    <Box>
      {videoUploaded ? (
        <Box flexDirection={'column'} h={'80%'}>
          <Flex direction='row' width={'100%'} >
            <Heading as="h1" size="5xl" mb={15} padding={'0.5em'}>
              Resultados
            </Heading>
            <Flex direction={'row'} alignItems={'center'} w='100%' padding={'2em'} borderWidth='0.2em' borderRadius='lg' borderColor={'black'} >
              <Image src={`data:image/jpeg;base64,${data.videoFrames[selectedIndex]}`} h={'20em'} padding={'0.2em'} />
              <Image src={`data:image/jpeg;base64,${data.processedFrames[selectedIndex]}`} h='12.5em' padding={'0.2em'} />
              <Box>
                <Text fontSize="lg" >
                  <b>Frame number</b>: {selectedIndex}
                </Text>
                <Text fontSize="lg">
                  <b>Total frame count</b>: {data.nFrames}
                </Text>
                <Text fontSize="lg">
                  <b>Fake %</b>: {data.predictions.data[selectedIndex].y * 100}%
                </Text>
              </Box>
            </Flex>
          </Flex>
          <Center>
            {/* Contenedor para la gráfica */}
            <div style={{ height: '18em', width: '90%'}}>
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
          </Center>

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
            <FormControl padding="0.5em">
              <InputGroup marginTop="1em" marginBottom="1em">
                <Input
                  type="file"
                  id="videoInput"
                  accept="video/mp4"
                  onChange={handleVideoUpload}
                  style={{ display: 'none' }} // Hide the default file input
                />
                <Button as="label" leftIcon={<IoMdVideocam />} htmlFor="videoInput" colorScheme="teal" variant="outline" size="md">
                  Sube un video
                </Button>
              </InputGroup>
            </FormControl>
          </Flex>

        )}
    </Box>
  );
};

export default BodyView;
