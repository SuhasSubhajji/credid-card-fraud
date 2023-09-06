/*
SQLyog Community Edition- MySQL GUI v6.07
Host - 5.0.27-community-nt : Database - creditcard
*********************************************************************
Server version : 5.0.27-community-nt
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

create database if not exists `creditcard`;

USE `creditcard`;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

/*Table structure for table `t1` */

DROP TABLE IF EXISTS `t1`;

CREATE TABLE `t1` (
  `username` varchar(55) default NULL,
  `password` varchar(55) default NULL,
  `email` varchar(55) default NULL,
  `phoneno` varchar(55) default NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `t1` */

insert  into `t1`(`username`,`password`,`email`,`phoneno`) values ('a','b','c','d'),('a','b','c','d');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
