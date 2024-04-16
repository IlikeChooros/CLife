#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <list>
#include <stddef.h>
#include <SFML/Graphics.hpp>

#include "namespaces.hpp"


START_NAMESPACE_UI

struct _ThreadData{
  // Wheter the Plotter is already killed, used for the keepAlive thread
  bool killed = false;
  // Wheter the keepAlive thread should keep the window alive
  bool keepAlive = false;
  // Mutex used for multithreading
  std::mutex mutex;
};

struct _PlotPoint{
  float x;
  float y;
};

struct _Plot{
  int x;
  int y;
  bool valid = true;
};

enum class DrawingPolicy{
  Point,
  LineConnected,
  LineDisconnected
};

enum class _DrawMethod{
  Single,
  Multiple
};

constexpr int _stateStrictValues = 1, _stateStrictRanges = 2;

using pltdata = std::list<_Plot>;
using data_t = std::list<_PlotPoint>;

class Plotter{

  typedef std::function<void(data_t*)> _callbackType;

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
  _DrawMethod _drawMethod;

  data_t _rawdata;
  pltdata _plotdata;
  sf::RenderWindow _window;

  _callbackType _callb;

  std::shared_ptr<_ThreadData> _threadData;
  std::thread _keepAliveThread;

  /*
  Should be run in a detached thread, keeps the window alive, handles
  only the window events
  */
  static void _keepAlive(Plotter* plotter);

  /*
  Draws the background, clears whole screen
  */
  void _drawBackground();

  /*
  Draws axis, data should be already prepared
  */
  void _drawAxis();

  /*
  Draws the data, by given drawing policy
  */
  void _drawData();
  void _drawPoint(_Plot& current);
  void _drawAllPoints();

  int _getNormalizedY(float y, bool policy = true);
  int _getNormalizedX(float x);
  _Plot _normalize(const _PlotPoint&, int, int);
  void _prepareData();

  bool _isStrict();
  bool _isDetached();

  public:
  Plotter(DrawingPolicy policy = DrawingPolicy::Point, std::size_t maxWidth = 1000, std::size_t maxHeight = 1000);
  ~Plotter();

  /**
   * @brief Prepare the plotter for drawing
  */
  void prepare(DrawingPolicy policy, std::size_t maxWidth, std::size_t maxHeight);

  /**
   * @brief Set the fixed values of the plotter (y-axis values)
  */
  void values(float min, float max);

  /**
   * @brief Set the fixed range of the plotter (x-axis range)
  */
  void range(float min, float max);

  /**
   * @brief Adds a point to the plotter
  */
  void add(const _PlotPoint& point);

  /**
   * @brief Add the vector data to the plotter
  */
  void add(const data_t& data);

  /**
   * @brief Set the data to the plotter
  */
  void set(const data_t& data);

  /**
   * @brief Force update the data normalizing it
  */
  void update();

  /**
   * @brief Add a callback to the plotter
  */
  void addCallback(_callbackType callback);

  /**
   * @brief Show one frame of the plotter
  */
  void show();

  /**
   * @brief Keeps the plotter alive, until the window is closed, 
   * creates a new detached thread to handle the window. To close the window
   * call the `close` method
   * @warning This function should be called only once, not thread safe
  */
  void keepAlive();

  /**
   * @brief Close the plotter, stops the loop and closes the window
  */
  void close();

  /**
   * @brief Open the plotter, and start the loop 
  */
  void open();

  /**
   * @brief Check if the plotter is closed
  */
  bool closed();
};


END_NAMESPACE