/*
 * (C) Copyright 2016-2018 Ben Karsin, Nodari Sitchinava
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>

#define RANGE 1000000

/* CPU Functions to create testData */
template<typename T>
void create_sorted_list(T* data, int size, int addVal) {
  for(int i=0; i<size; i++) {
//    data[i].key = i+addVal;
//    data[i].val = i-addVal;
  }
}
/*
template<typename T>
void create_random_sorted_list(T* data, int size) {
//  srand(time(NULL));
  for(int i=0; i<size; i++) {
    data[i] = {.key = rand()%RANGE, .val= rand()%RANGE};
  }
  std::sort(data, data+size);
}
*/

template<typename T>
void create_random_list(T* data, int size, int min) {
  srand(time(NULL));
  long temp;
//printf("size:%d\n", size);
  for(int i=0; i<size; i++) {
//    data[i].key = rand()%RANGE + min;
//    data[i].val = rand()%RANGE + min;
    data[i] = (rand()%RANGE) + min + 1;
    temp = rand()%RANGE + min + 1;
    data[i] += (temp<<32);
  }
}
