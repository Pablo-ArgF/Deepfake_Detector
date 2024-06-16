import React, { useState } from 'react';
import {
  Heading,
  Text,
  FormControl,
  Flex,
  Input,
  InputGroup,
  Button,
  Box,
  Spinner,
  IconButton
} from '@chakra-ui/react';
import { IoMdVideocam } from 'react-icons/io';
import { ArrowBackIcon } from '@chakra-ui/icons';
import CNNVideoDashboard from './CNNVideoDashboard';
import RNNVideoDashboard from './RNNVideoDashboard';  // Import RNNVideoDashboard

const BodyView = () => {
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [RNNdata, setRNNData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(1);
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [RNNloading, setRNNLoading] = useState(true);
  const [useRNN, setUseRNN] = useState(false);  // State to control which dashboard to use

  const handleVideoUpload = async (event) => {
    setError('');
    const file = event.target.files[0];
    if (!file) {
      setError('Por favor, selecciona un archivo de video.');
      return;
    }
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('video', file);

      // Set a 10-minute timeout (600,000 milliseconds)
      const controller = new AbortController();
      var timeoutId = setTimeout(() => controller.abort(), 600000);

      fetch('http://156.35.163.188/api/predict', {
        method: 'POST',
        body: formData,
        headers: {
          'enctype': 'multipart/form-data'
        },
        signal: controller.signal // Attach the signal to the fetch request
      }).then(async response => {
          var CNNdata = await response.json();
          setData(CNNdata);
          setLoading(false);
          setVideoUploaded(true);
          // Clear the timeout if the request completes before the timeout fires
          clearTimeout(timeoutId);

           //Start the prediction by sequences
          var timeoutIdRNN = setTimeout(() => controller.abort(), 600000);
    
          fetch('http://156.35.163.188/api/predict/sequences', {
            method: 'POST',
            body: formData,
            headers: {
              'enctype': 'multipart/form-data'
            },
            signal: controller.signal // Attach the signal to the fetch request
          }).then(async RNNresponse =>{
              const RNNdata = await RNNresponse.json();
              setRNNData(RNNdata);
              setRNNLoading(false);
              // Clear the timeout if the request completes before the timeout fires
              clearTimeout(timeoutIdRNN);
          });
      });
    } catch (error) {
      setError('Error prediciendo DeepFakes: ' + error);
      setLoading(false);
      setVideoUploaded(false);
      setData(null);
    }
  };

  return (
    <Box>
      <Flex direction='row' width={'100%'} justifyContent={'space-evenly'} visibility={videoUploaded? 'visible': 'hidden'}>
      <IconButton
            colorScheme='grey'
            padding={'0.7em'}
            borderRadius={'1em'}
            width={'3em'}
            height={'3em'}
            marginLeft={'1em'}
            marginRight={'1.5em'}
            marginTop={'1em'}
            icon={<ArrowBackIcon boxSize={'2.8em'}/>}
            onClick={() => {
              setVideoUploaded(false); 
              setData(null);
              setLoading(false);
            }}
          />
        <Heading w='100%' as="h1" size="5xl" padding={'0.1em'} flexGrow={4}>
          {useRNN ? 'Análisis de secuencias (RNN)' :'Análisis frame por frame (CNN)' }
        </Heading>
        <Button 
          marginRight={'1em'}
          marginTop={'1em'}
          cornerRadius={'1em'}
          backgroundColor={'black'}
          textColor={'white'}
          fontSize={'1em'}
          padding={'0.7em'}
          colorScheme='blue'
          w={'15em'}
          h={'3em'}
          onClick={() => setUseRNN(!useRNN)}  // Toggle the state when button is clicked
        >
          {useRNN ? 'Analizar frame por frame' : 'Analizar secuencias'}
        </Button>
      </Flex>
      {videoUploaded ? (
        useRNN ? (
          <RNNVideoDashboard 
            setData={setRNNData}
            setLoading={setRNNLoading}
            loading = {RNNloading}
            data={RNNdata} 
            setSelectedIndex={setSelectedIndex} 
            selectedIndex={selectedIndex} 
          />
        ) : (
          <CNNVideoDashboard 
            setVideoUploaded={setVideoUploaded} 
            setData={setData}
            setLoading={setLoading}
            data={data} 
            setSelectedIndex={setSelectedIndex} 
            selectedIndex={selectedIndex} 
          />
        )
      ) : (
        <Flex direction="column" align="center" padding="2em">
          <Heading as="h1" size="5xl" mb={15}>
            Detección de DeepFakes
          </Heading>
          <Text textAlign="justify" width="45%">
            El incipiente uso de las inteligencias artificiales ha hecho que la suplantación de identidad a través de <b>DeepFakes</b>, contenido sintético generado por algoritmos de inteligencia artificial que combinan y superponen imágenes y vídeos existentes para crear uno nuevo.<br/><br/>
            El nivel de realismo que se obtiene utilizando algoritmos de DeepFake actuales supone un <b>riesgo a la sociedad</b>. Desde fake news hasta suplantaciones de identidad, la aparición de estas inteligencias artificiales han hecho que el contenido que consumimos cada vez sea menos identificable<br/><br/>
            Esta herramienta intenta ayudar en la identificación de este tipo de material sintético mediante un análisis aplicando modelos de predicción de DeepFakes.
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
                cornerRadius='0.5em'
              />
              <Button as="label" leftIcon={<IoMdVideocam color='white'/>} htmlFor="videoInput" backgroundColor={'black'} textColor={'white'} padding={'1.1em'} fontSize={'1.1em'}>
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
