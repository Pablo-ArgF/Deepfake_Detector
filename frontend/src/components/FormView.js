// FormView.js
import React, { useState } from 'react';
import {
  Center,
  Heading,
  Text,
  FormControl,
  FormLabel,
  Input,
  Button,
  Flex,
  InputGroup,
  InputLeftElement,
} from '@chakra-ui/react';
import { IoMdVideocam } from 'react-icons/io';
import { PiGraphFill } from 'react-icons/pi';

const FormView = () => {
  const [error, setError] = useState('');

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];

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
    } catch (error) {
      console.error('Error prediciendo DeepFakes:', error);
    }
  };

  return (
    <Flex direction="column" align="center" padding="2em">
      <Center>
        <Heading as="h1" size="5xl" mb={15}>
          Detección de DeepFakes
        </Heading>
      </Center>
      <Text textAlign="center" width="50%">
        El incipiente uso de las inteligencias artificiales ha hecho que la suplantación de identidad a través de deepFakes (contenido sintético generado por algoritmos de inteligencia artificial que combinan y superponen imágenes y vídeos existentes para crear uno nuevo, a menudo reemplazando la apariencia de una persona con la de otra). La herramienta que estás a punto de probar intenta ayudar en la identificación de este tipo de material sintético. Prueba a subir un video a continuación para comprobar si contiene DeepFakes o no.
      </Text>
      <Center>
        <FormControl padding="0.5em">
          <FormLabel htmlFor="videoInput">Subir video:</FormLabel>
          <InputGroup marginTop="2em" marginBottom="2em">
            <InputLeftElement pointerEvents="none">
              <IoMdVideocam size="1.5em" />
            </InputLeftElement>
            <Input
              type="file"
              id="videoInput"
              accept="video/*"
              onChange={handleVideoUpload}
            />
          </InputGroup>
          {error && (
            <Text color="red" fontSize="sm" marginBottom="1em">
              {error}
            </Text>
          )}
          <Button
            leftIcon={<PiGraphFill />}
            mt={4}
            colorScheme="black"
            padding="1em"
          >
            Detectar DeepFakes
          </Button>
        </FormControl>
      </Center>
    </Flex>
  );
};

export default FormView;
