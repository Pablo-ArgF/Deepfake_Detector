import React from 'react';
import {Text, Flex, Image } from '@chakra-ui/react';

const Navbar = ({setView}) => {
  return (
    <Flex
      as="nav"
      align="left"
      justify="space-between"
      wrap="wrap"
      padding="1rem"
      bg="gray" // Dark gray background
      color="white">
      
      <Image width="18em" src="./Logo Universidad de Oviedo.png" alt="Logo Universidad de Oviedo" onClick={() => setView('BodyView')} cursor="pointer"  />
      <Text fontSize="lg" fontWeight="bold" alignSelf={'center'}>
        DeepFake Detection Final Degree project
      </Text> 

      <Flex
        direction="row"
        mt={{ base: 4, md: 0 }}
        padding="2em">
        <Text mr={50} cursor="pointer" onClick={() => setView('About')} >
          About
        </Text>
        <Text cursor="pointer" onClick={() => setView('ModelDetails')}>
          Model
        </Text>
      </Flex>
    </Flex>
  );
};

export default Navbar;
