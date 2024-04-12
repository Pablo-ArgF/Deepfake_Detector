import React from 'react';
import { Box, Text, Flex, Image, useDisclosure } from '@chakra-ui/react';
import { CloseIcon, HamburgerIcon } from '@chakra-ui/icons';

const Navbar = () => {
  return (
    <Flex
      as="nav"
      align="left"
      justify="space-between"
      wrap="wrap"
      padding="1rem"
      bg="gray" // Dark gray background
      color="white"
      width="100em"
    >
      
      <Image width="10%" src="./Logo Universidad de Oviedo.png" alt="Logo Universidad de Oviedo" />
      <Text fontSize="lg" fontWeight="bold">
        DeepFake Detection Final Degree project
      </Text> 

      <Flex
        direction="row"
        mt={{ base: 4, md: 0 }}
        padding="2em">
        <Text mr={4} cursor="pointer">
          About
        </Text>
        <Text cursor="pointer">Model</Text>
      </Flex>
    </Flex>
  );
};

export default Navbar;
