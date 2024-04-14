#pragma once

#include <vector>
#include <stddef.h>
#include <SFML/Graphics.hpp>

#include "namespaces.hpp"


START_NAMESPACE_UI

struct _PlotPoint{
  float x;
  float y;
};

struct _Plot{
  int x;
  int y;
};

enum class DrawingPolicy{
  Point,
  LineConnected
};

constexpr int _stateStrictValues = 1;

using pltdata = std::vector<_Plot>;
using _rawdatatype = std::vector<_PlotPoint>;

class Plotter{

  typedef std::function<void(pltdata&)> _callbackType;

  std::size_t _maxWidth;
  std::size_t _maxHeight;
  
  int _xAxisPos;
  int _yAxisPos;

  float _maxValue;
  float _minValue;

  float _minRange;
  float _maxRange;

  int _state;
  bool _forceUpdate;

  DrawingPolicy _policy;

  _rawdatatype _rawdata;
  pltdata _plotdata;
  sf::RenderWindow _window;

  _callbackType _callb;

  void _drawBackground();
  void _drawAxis();
  void _drawData();
  void _drawPoint(_Plot& current, _Plot& prev);

  int _getNormalizedY(float y, bool policy = true);
  int _getNormalizedX(float x);
  _Plot _normalize(const _PlotPoint&);
  void _prepareData();

  bool _isStrict();

  public:
  Plotter(DrawingPolicy policy = DrawingPolicy::Point, std::size_t maxWidth = 1000, std::size_t maxHeight = 1000);

  void prepare(DrawingPolicy policy, std::size_t maxWidth, std::size_t maxHeight);
  void values(float min, float max);
  void range(float min, float max);

  void add(const _PlotPoint& point);
  void add(const _rawdatatype& data);
  void update();

  void addCallback(_callbackType callback);

  void open();
};


END_NAMESPACE