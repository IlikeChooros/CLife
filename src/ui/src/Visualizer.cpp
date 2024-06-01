#include <ui/Visualizer.hpp>

START_NAMESPACE_UI

void Visualizer::update(const _DataType &data)
{
  _data.push_back(data);
}

void ConsoleVisualizer::visualize()
{
  if (_data.empty())
  {
    return;
  }
  auto data = _data.back();
  std::cout << "Batch: " << data.batch << " Loss: " << data.loss << " Time: " << data.time << std::endl;
}

GraphVisualizer::GraphVisualizer() : _plotter(DrawingPolicy::LineConnected, 800, 800)
{
  _plotter.values(0, 1);
  _plotter.keepAlive();
}

void GraphVisualizer::update(const _DataType &data)
{
  if (_plotter.closed())
  {
    return;
  }
  _plotter.add({(float)data.time, (float)data.loss});
}

void GraphVisualizer::visualize()
{
  if (_plotter.closed())
  {
    return;
  }
  _plotter.show();
}

END_NAMESPACE