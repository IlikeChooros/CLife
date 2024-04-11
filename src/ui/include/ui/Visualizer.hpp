#pragma once

#include <iostream>


#include "namespaces.hpp"

START_NAMESPACE_UI

template<typename _DataType>
class Visualizer{
  public:
  Visualizer() = default;

  virtual void visualize() = 0;
  virtual void update(const _DataType& data);

  protected:
  _DataType _data;
};

template<typename _DataType>
void Visualizer<_DataType>::update(const _DataType& data){
  _data = data;
}

class ConsoleVisualizer : public Visualizer<std::string>{
  public:
  ConsoleVisualizer() = default;

  void visualize() override;
};

struct _GraphData{
  double loss; 
  double accuracy;
};

class GraphVisualizer : public Visualizer<_GraphData>{
  public:
  GraphVisualizer() = default;

  void visualize() override;
}; 

END_NAMESPACE