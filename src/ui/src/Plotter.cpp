#include <ui/Plotter.hpp>

START_NAMESPACE_UI

constexpr float _MIN = -0x7fffffff, _MAX = 0x7fffffff;
constexpr int axisThickness = 2, pointRadius = 4;

constexpr bool _checkState(int state, int value)
{
  return (state & value) == value;
}

constexpr _DrawMethod getMethod(DrawingPolicy policy)
{
  switch (policy)
  {
  case DrawingPolicy::Point:
    return _DrawMethod::Single;
  default:
    return _DrawMethod::Multiple;
  }
}

void do_nothing(data_t *) { return; }

Plotter::Plotter(
    DrawingPolicy policy,
    std::size_t maxWidth,
    std::size_t maxHeight) : _maxWidth(maxWidth), _maxHeight(maxHeight), _xAxisPos(maxHeight - axisThickness), _yAxisPos(0),
                             _maxValue(_MIN), _minValue(_MAX), _minRange(_MAX), _maxRange(_MIN),
                             _state(0), _forceUpdate(true), _policy(policy), _drawMethod(getMethod(policy)), _plotdata(),
                             _window(sf::VideoMode(maxWidth, maxHeight), "Plotter"), _callb(do_nothing),
                             _threadData(new _ThreadData)
{
}

Plotter::~Plotter()
{
  close();
  if (_keepAliveThread.joinable())
  {
    _keepAliveThread.join();
  }
}

void Plotter::prepare(DrawingPolicy policy, std::size_t maxWidth, std::size_t maxHeight)
{
  _policy = policy;
  _maxWidth = maxWidth;
  _maxHeight = maxHeight;
  _drawMethod = getMethod(policy);

  _window.create(sf::VideoMode(maxWidth, maxHeight), "Plotter");
}

void Plotter::values(float min, float max)
{
  _minValue = min;
  _maxValue = max;
  _state |= _state ^ _stateStrictValues;
}

void Plotter::range(float min, float max)
{
  _minRange = min;
  _maxRange = max;
  _state |= _state ^ _stateStrictRanges;
}

void Plotter::add(const _PlotPoint &point)
{
  _rawdata.push_back(point);
}

void Plotter::add(const data_t &data)
{
  _rawdata.insert(_rawdata.end(), data.begin(), data.end());
}

void Plotter::set(const data_t &data)
{
  _rawdata = data;
}

void Plotter::update()
{
  _forceUpdate = true;
}

void Plotter::addCallback(std::function<void(data_t *)> callb)
{
  _callb = callb;
}

void Plotter::show()
{
  // Lock the mutex, and set the window to active, since
  // the window is shared between threads
  std::lock_guard<std::mutex> lock(_threadData->mutex);

  // Send a signal to update the frame
  if (_isDetached())
  {
    _threadData->update = true;
    return;
  }

  _window.setActive();

  _show();
  // Set the window to inactive, and unlock the mutex
  _window.setActive(false);
}

void Plotter::_keepAlive(Plotter *plotter)
{

  std::shared_ptr<_ThreadData> data(plotter->_threadData);
  {
    std::lock_guard<std::mutex> lock(data->mutex);
    // If the window is already detached, return
    if (plotter->_isDetached())
    {
      return;
    }
    // Set the keepAlive flag to true (_isDetached() will return true)
    data->keepAlive = true;
  }

  // Set this thread as active for drawing
  plotter->_window.setActive();

  while (true)
  {
    data->mutex.lock();

    // Check if the window is already killed, or closed
    if (data->killed || !plotter->_window.isOpen())
    {
      data->mutex.unlock();
      break;
    }

    if (data->update)
    {
      plotter->_show();
      data->update = false;
    }

    sf::Event e;
    if (plotter->_window.pollEvent(e))
    {
      if (sf::Event::Closed == e.type)
      {
        plotter->_window.close();
      }
    }

    data->mutex.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Deactivate the window from this thread
  plotter->_window.setActive(false);
}

void Plotter::keepAlive()
{
  if (_isDetached())
  {
    return;
  }
  _window.setActive(false);
  _keepAliveThread = std::thread(_keepAlive, this);
  _keepAliveThread.detach();
}

void Plotter::close()
{
  std::lock_guard<std::mutex> lock(_threadData->mutex);
  _threadData->killed = true;

  _window.setActive();
  _window.close();
}

void Plotter::open()
{

  std::lock_guard<std::mutex> lock(_threadData->mutex);
  if (_isDetached())
  {
    return;
  }

  _window.setActive();
  _window.display();

  sf::Clock timer;

  constexpr int FPS = 60, _FT = 1000 / FPS;

  while (_window.isOpen())
  {
    sf::Event e;
    if (_window.pollEvent(e))
    {
      switch (e.type)
      {
      case sf::Event::Closed:
        _window.close();
        break;
      default:
        break;
      }
    }

    _callb(&_rawdata);

    // 60 FPS
    if (timer.getElapsedTime().asMilliseconds() < _FT)
    {
      continue;
    }
    timer.restart();

    _prepareData();
    _drawBackground();
    _drawAxis();
    _drawData();
    _window.display();
  }

  _window.setActive(false);
}

bool Plotter::closed()
{
  return !_window.isOpen();
}

void Plotter::_show()
{
  _forceUpdate = true;
  _prepareData();
  _drawBackground();
  _drawAxis();
  _drawData();
  _window.display();
}

bool Plotter::_isStrict()
{
  return _checkState(_state, _stateStrictValues) && _checkState(_state, _stateStrictRanges);
}

bool Plotter::_isDetached()
{
  return _threadData->keepAlive;
}

int Plotter::_getNormalizedY(float y, bool policyEffect)
{
  int ret = static_cast<int>((1 - (y - _minValue) / (_maxValue - _minValue)) * _maxHeight);
  if (policyEffect && _policy == DrawingPolicy::Point)
  {
    ret -= pointRadius;
  }
  return ret;
}

int Plotter::_getNormalizedX(float x)
{
  return static_cast<int>((x - _minRange) / (_maxRange - _minRange) * _maxWidth);
}

_Plot Plotter::_normalize(const _PlotPoint &point, int xDelta, int yDelta)
{
  bool valid = point.x >= _minRange && point.x <= _maxRange && point.y >= _minValue && point.y <= _maxValue;
  return {
      _getNormalizedX(point.x) + xDelta,
      _getNormalizedY(point.y) + yDelta,
      valid};
}

void Plotter::_drawBackground()
{
  _window.clear(sf::Color(0, 12, 24));
}

void Plotter::_drawAxis()
{
  using namespace sf;
  RectangleShape xA(Vector2f(_maxWidth, axisThickness));
  RectangleShape yA(Vector2f(axisThickness, _maxHeight));

  auto color = sf::Color(49, 58, 77);

  xA.setFillColor(color);
  yA.setFillColor(color);

  xA.setPosition(0, _xAxisPos);
  yA.setPosition(_yAxisPos, 0);

  _window.draw(xA);
  _window.draw(yA);
}

void Plotter::_drawPoint(_Plot &current)
{
  using namespace sf;
  auto color = Color(45, 68, 124);

  switch (_policy)
  {
  case DrawingPolicy::Point:
  {
    CircleShape p(pointRadius);
    p.setPosition(current.x, current.y);
    p.setFillColor(color);
    _window.draw(p);
  }
  break;

  default:
    break;
  }
}

void Plotter::_drawAllPoints()
{
  if (_plotdata.size() < 2)
  {
    return;
  }

  using namespace sf;

  auto color = Color(45, 68, 124);

  PrimitiveType primitive;

  switch (_policy)
  {
  case DrawingPolicy::LineConnected:
    primitive = PrimitiveType::LineStrip;
    break;
  case DrawingPolicy::LineDisconnected:
    primitive = PrimitiveType::Lines;
    break;
  default:
    break;
  }

  VertexArray line(primitive, _plotdata.size());

  int i = 0;
  for (auto it = _plotdata.begin(); it != _plotdata.end(); it++, i++)
  {
    line[i].position = Vector2f(it->x, it->y);
    line[i].color = color;
  }
  _window.draw(line);
}

void Plotter::_prepareData()
{
  if (!_forceUpdate)
  {
    return;
  }

  if (!_isStrict())
  {
    for (auto it = _rawdata.begin(); it != _rawdata.end(); it++)
    {
      if (!_checkState(_state, _stateStrictValues))
      {
        _maxValue = std::max(_maxValue, it->y);
        _minValue = std::min(_minValue, it->y);
      }
      if (!_checkState(_state, _stateStrictRanges))
      {
        _minRange = std::min(_minRange, it->x);
        _maxRange = std::max(_maxRange, it->x);
      }
    }
  }

  if (_minValue == _maxValue)
  {
    _minValue -= 1;
    _maxValue += 1;
  }

  if (_minRange == _maxRange)
  {
    _minRange -= 1;
    _maxRange += 1;
  }

  int xDelta = 0, yDelta = 0;

  _xAxisPos = _getNormalizedY(0, false);
  _yAxisPos = _getNormalizedX(0);

  // check if the axis is out of bounds
  if (_minRange > 0)
  {
    // axis should be negative, must set it to the left
    // the graph is to the right of the Y axis
    xDelta = -_yAxisPos;
    _yAxisPos = 0;
    _minRange = 0; // extend the range
  }

  if (_maxRange < 0)
  {
    // axis should be positive, must set it to the right
    // the graph is to the left of the Y axis
    xDelta = _yAxisPos - _maxWidth; // must be positive
    _yAxisPos = static_cast<int>(_maxWidth) - axisThickness;
    _maxRange = 0; // extend the range
  }

  if (_minValue > 0)
  {
    // axis should be negative, must set it to the top
    // (the graph is below the X axis)
    yDelta = _maxHeight - _xAxisPos; // must be negative
    _xAxisPos = static_cast<int>(_maxHeight) - axisThickness;
    _minValue = 0; // extend the range
  }

  if (_maxValue < 0)
  {
    // axis should be positive, must set it to the bottom
    // (the graph is above the X axis)
    yDelta = -_xAxisPos;
    _xAxisPos = 0;
    _maxValue = 0; // extend the range
  }

  _xAxisPos = std::min(_xAxisPos, static_cast<int>(_maxHeight) - axisThickness);
  _yAxisPos = std::min(_yAxisPos, static_cast<int>(_maxWidth) - axisThickness);

  _plotdata.clear();

  for (auto it = _rawdata.begin(); it != _rawdata.end(); it++)
  {
    _plotdata.push_back(_normalize(*it, xDelta, yDelta));
  }

  _forceUpdate = false;
}

void Plotter::_drawData()
{
  if (_plotdata.empty())
  {
    return;
  }

  if (_drawMethod == _DrawMethod::Multiple)
  {
    _drawAllPoints();
  }
  else
  {
    for (auto &point : _plotdata)
    {
      _drawPoint(point);
    }
  }
}

END_NAMESPACE