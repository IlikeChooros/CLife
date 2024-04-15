#pragma once

#include <iostream>
#include <list>

#include "Plotter.hpp"
#include "namespaces.hpp"


START_NAMESPACE_UI

struct _DataType{
  int batch;
  double loss;
  int64_t time;
};

class Visualizer{
  public:
  Visualizer() = default;
  virtual ~Visualizer() = default;

  virtual void visualize() {return;}
  virtual void update(const _DataType& data);

  protected:
  std::list<_DataType> _data;
};

void Visualizer::update(const _DataType& data){
  _data.push_back(data);
}

class ConsoleVisualizer : public Visualizer{
  public:
  ConsoleVisualizer() = default;

  void visualize() override;
};

class GraphVisualizer : public Visualizer{
  Plotter _plotter;
  public:
  GraphVisualizer();
  ~GraphVisualizer() override;

  void visualize() override;
  void update(const _DataType& data) override;
}; 

END_NAMESPACE