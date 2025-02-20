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
      bg="#C2C2C2" // Dark gray background
      color="black">
      
      <Image width="18em" src="./Logo Universidad de Oviedo.png" alt="Logo Universidad de Oviedo" onClick={() => setView('BodyView')} cursor="pointer"  />
      <Text fontSize="1.3em" fontWeight="bold" alignSelf={'center'}>
        DeepFake Detection
      </Text> 

      <Flex
        direction="row"
        mt={{ base: 4, md: 0 }}
        padding="2em">
        <Text mr={50} fontSize="1.1em" cursor="pointer" fontWeight={'bold'} onClick={() => setView('About')} >
          About
        </Text>
        <Text cursor="pointer" fontSize="1.1em" fontWeight={'bold'}  onClick={() => setView('ModelDetails')}>
          Model
        </Text>
      </Flex>
    </Flex>
  );
};

export default Navbar;
