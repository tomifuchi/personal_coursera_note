(window.webpackJsonp=window.webpackJsonp||[]).push([[7,196],{"+M5F":function(module,e,t){"use strict";t.d(e,"a",function(){return f});var n=t("S+eF"),r=t.n(n),o=t("JAlB"),i=t("wDa7"),a=t("CA+Q"),s=t("ML/G"),u=t("GnyC"),c=function receiveDeadlines(e,t){var n=t.deadlines,r=n.isEnabled,o=n.moduleDeadlines;r?e.dispatch("LOAD_COURSE_DEADLINES",{moduleDeadlines:o}):e.dispatch("DISABLE_DEADLINES")},d=function enableDeadlines(e){var t=e.getStore("CourseStore").getCourseId();return i.a.sendStartTime(!0,t).fail(function(e){throw e}).then(a.a).then(function(t){var n=t.elements,r=n[0].start;return s.a.pushV2(["open_course_home.welcome.emit.course_deadline_set",{first_week_due_time:r}]),e.executeAction(c,{deadlines:n[0]})})},l=function setDeadlinesIfEligible(e){var t=e.getStore("CourseStore"),n=e.getStore("CourseScheduleStore"),i=e.getStore("ProgressStore"),a=Object(u.a)(t,n,i),s=t.getCourseId(),c=e.getStore("CourseMembershipStore").isEnrolled(),l=e.getStore("SessionStore");return Object(o.a)(s)||l.isSessionsCourse()||1!==a||!c?r()():e.executeAction(d,{})},f=function loadCourseDeadlines(e,t){var n=t.userId,s=e.getStore("CourseStore").getCourseId(),u=e.getStore("CourseMembershipStore").isEnrolled(),d=e.getStore("SessionStore"),f=e.getStore("CourseStore").isReal();if(!u||!n||Object(o.a)(s))return r()();if(d.isSessionsEnabled()){if(d.isEnrolled()){var p=d.getSession(),g={moduleDeadlines:p.moduleDeadlines};g.itemDeadlines=p.itemDeadlines,e.dispatch("LOAD_COURSE_DEADLINES",g)}return r()()}return i.a.getStartTime(s).then(a.a).then(function(t){var n,r=t.elements[0];return r?e.executeAction(c,{deadlines:r}):e.executeAction(l,{})})},p=function disableDeadlines(e){var t=e.getStore("CourseStore").getCourseId();return i.a.disableDeadlines(t).then(function(){return e.dispatch("DISABLE_DEADLINES")}).fail(function(e){throw e})},g=function resetDeadlines(e,t){var n=t.userId,r=e.getStore("CourseStore").getCourseId();return i.a.resetDeadlines(r).then(function(){return e.executeAction(f,{userId:n})}).fail(function(e){throw e})}},"+MHB":function(module,e,t){"use strict";var n=t("welz"),r=t.n(n),o=t("tDXo"),i=t("+MK6"),a=t("P4G6"),s=new o.a("[data-js=origami]");s.router.on("route",function(){Object(a.b)()}),s.enableRR=function(){var e=s.router;s.router=i.a,s.router.on("routerWillLeave",function(t){e.unsavedWarningModal(t)}),r.a.history={}},s.on("deprecatedUsage",a.a),e.a=s},"+MK6":function(module,e,t){"use strict";var n=t("RIqP"),r=t.n(n),o=t("VbXa"),i=t.n(o),a=t("oYk5"),s=t.n(a),u=t("F/us"),c=t.n(u),d=t("BVC1"),l=t("yiR1"),f=t.n(l),p=t("47m/"),g=t.n(p),h=t("Af4h"),v=t.n(h),m=t("q1tI"),S=t.n(m),y=t("j4LP"),O,w=function App(e){var t=e.children;return S.a.createElement("div",null,S.a.createElement("div",{"data-js":"origami"}),t)};w.propTypes={children:S.a.PropTypes.node};var b=function(e){function NotFoundRedirect(){return e.apply(this,arguments)||this}i()(NotFoundRedirect,e);var t=NotFoundRedirect.prototype;return t.componentWillMount=function componentWillMount(){O.trigger("error",404)},t.render=function render(){return null},NotFoundRedirect}(S.a.Component),E=function legacyRouteHandler(e,t,n){var r,o;return o=r=function(r){function LegacyRouteHandler(){for(var e,t=arguments.length,n=new Array(t),o=0;o<t;o++)n[o]=arguments[o];return(e=r.call.apply(r,[this].concat(n))||this).state={userConfirmed:!1},e.routerWillLeave=function(t){if(e.state.userConfirmed)return!0;if(t)return e.setState({nextLocation:t}),O.trigger("routerWillLeave",function(t){t?e.setState({userConfirmed:t},function(){e.context.router.push(e.state.nextLocation)}):e.setState({nextLocation:null,userConfirmed:!1})}),!1},e}i()(LegacyRouteHandler,r);var o=LegacyRouteHandler.prototype;return o.componentDidMount=function componentDidMount(){this.executeLegacyRoute()},o.componentDidUpdate=function componentDidUpdate(e){this.props.location.key!==e.location.key&&this.executeLegacyRoute()},o.executeLegacyRoute=function executeLegacyRoute(){var r=t(e,this.context.router.params);n.apply(this,r.concat(this.context.router.location.query)),this.context.router.setRouteLeaveHook(this.props.route,this.routerWillLeave)},o.render=function render(){return S.a.createElement("div",null)},LegacyRouteHandler}(S.a.Component),r.propTypes={location:S.a.PropTypes.object.isRequired,route:S.a.PropTypes.object.isRequired},r.contextTypes={router:S.a.PropTypes.object.isRequired},o},I;O=new(function(){function ReactRouterAdapter(){this.routes={},this.nativeRoutes=[],this.visitedFragments=[],f.a.emitter(this)}var e=ReactRouterAdapter.prototype;return e.addRoutes=function addRoutes(e,t){t?Object.assign(this.routes,c()(e).mapObject(t)):Object.assign(this.routes,e)},e.addNativeRoutes=function addNativeRoutes(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:[];this.nativeRoutes.push(e)},e.getLegacyRoutes=function getLegacyRoutes(e){var t=this;return c()(this.routes).map(function(n,r){return r?S.a.createElement(g.a,{key:r,path:r,component:E(r,t.getLegacyRouteParams,n)}):S.a.createElement(v.a,{key:"index:".concat(e),component:E(r,t.getLegacyRouteParams,n)})})},e.getRoutes=function getRoutes(e){if(!this.getLegacyRoutes(e).length&&this.nativeRoutes.length)return this.nativeRoutes;var t=[].concat(r()(this.getLegacyRoutes(e)),r()(this.nativeRoutes),[S.a.createElement(g.a,{path:"*",key:"notFoundRoute",component:b})]);return S.a.createElement(g.a,{path:e,component:w},t)},e.getLegacyRouteParams=function getLegacyRouteParams(e,t){var n=/\:\w*/g,r=e.match(n);return r?c()(r).chain().compact().invoke("slice",1).reduce(function(e,n){return e.concat(t[n])},[]).value():[]},e.setupLinks=function setupLinks(e,t){this.routePrefix=e,this.linkSelector=t,s()("body").on("click","a"+this.linkSelector,function(e){e.ctrlKey||e.metaKey||(e.preventDefault(),this.navigate(s()(e.currentTarget).attr("href"),!0))}.bind(this))},e.navigate=function navigate(e,t){var n=e;0!==n.indexOf(this.routePrefix)&&(n=d.a.join(this.routePrefix,n)),t&&t.replace||this.addFragmentToVisited(n),!0===t||t&&!0===t.trigger?t&&t.replace?this.rr.replace(n):this.rr.push(n):t&&t.replace?window.history.replaceState(null,null,n):window.history.pushState(null,null,n)},e.navigateTo=function navigateTo(e,t){if(!e&&""!==e)throw new Error("router.navigateTo received no href");this.navigate(e.replace(this.routePrefix,""),t)},e.addFragmentToVisited=function addFragmentToVisited(e){this.visitedFragments.push(e)},e.start=function start(e,t){var n=this,r=Object(y.a)("rendered-content");Object(y.b)(this.getRoutes(e),r).then(function(e){n.rr=e})},ReactRouterAdapter}()),e.a=O},"5Ujo":function(module,e,t){"use strict";t.d(e,"a",function(){return i});var n=t("S+eF"),r=t.n(n),o=t("Vu1r"),i=function loadCoursePresentGrade(e,t){var n=t.userId,i=t.courseId;if(e.getStore("CoursePresentGradeStore").hasLoaded())return r()();return n?r()(o.a.getPresentGrade({userId:n,courseId:i})).then(function(t){var n=t.elements[0];e.dispatch("LOAD_COURSE_PRESENT_GRADE",{presentGrade:n})}).fail(function(t){e.dispatch("LOAD_COURSE_PRESENT_GRADE_FAIL",{})}):(e.dispatch("LOAD_COURSE_PRESENT_GRADE_FAIL",{}),r()())}},"7MHW":function(module,e,t){"use strict";t.d(e,"a",function(){return provideRequestCountryCode});var n=t("VbXa"),r=t.n(n),o=t("17x9"),i=t.n(o),a=t("q1tI"),s=t.n(a),u=t("w965");function provideRequestCountryCode(e){var t=e.displayName||e.name,n=function(t){function RequestCountryCodeContextProvider(){return t.apply(this,arguments)||this}r()(RequestCountryCodeContextProvider,t);var n=RequestCountryCodeContextProvider.prototype;return n.getChildContext=function getChildContext(){return{requestCountryCode:u.a}},n.render=function render(){return s.a.createElement(e,this.props)},RequestCountryCodeContextProvider}(s.a.Component);return n.displayName=t+"RequestCountryCodeContextProvider",n.childContextTypes={requestCountryCode:i.a.string},n}},"8iQH":function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("fw5G"),i=t.n(o),a=t("DnuM"),s=t("15pW");e.a=function(e){var t=new a.a("/api/courses.v1",{type:"rest"}),n=(new i.a).addQueryParam("q","slug").addQueryParam("slug",e).addQueryParam("fields","certificates").addQueryParam("showHidden",!0);return r()(t.get(n.toString())).then(function(t){if("notFound"===t.errorCode)return null;var n=t.elements[0],r=n.id,o=n.certificates;return s.d.courseId=r,s.d.courseSlug=e,{courseId:r,courseCertificates:o}})}},AbQT:function(module,exports){function webpackEmptyContext(e){var t=new Error("Cannot find module '"+e+"'");throw t.code="MODULE_NOT_FOUND",t}webpackEmptyContext.keys=function(){return[]},webpackEmptyContext.resolve=webpackEmptyContext,module.exports=webpackEmptyContext,webpackEmptyContext.id="AbQT"},BYIE:function(module,e,t){"use strict";var n=t("sArp"),r=t("CA+Q");e.a=function(e){if(!e)throw new Error("`courseId` is required to get course schedule.");return Object(n.a)(e).then(r.a).then(function(e){var t;return e.elements[0].defaultSchedule.periods})}},Bgx3:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("8iQH"),i=t("IG+7");e.a=function(e,t){if(e.getStore(i.a).haveCourseIdentifiersLoaded())return r()();if(!t)throw new Error("Missing courseSlug");return Object(o.a)(t).then(function(n){var r=n.courseId,o=n.courseCertificates;if(!r)throw new Error("Missing courseId");return e.dispatch("SET_COURSE_IDENTIFIERS",{courseId:r,courseSlug:t,courseCertificates:o}),{courseId:r,courseSlug:t,courseCertificates:o}}).catch(function(n){console.error("Error getting courseId and courseCertificates from courseSlug: ".concat(t,": "),n,n.stack);var o="",i=[];return e.dispatch("SET_COURSE_IDENTIFIERS",{courseId:"",courseSlug:t,courseCertificates:i}),r()({courseId:"",courseSlug:t,courseCertificates:i})})}},D87r:function(module,e,t){"use strict";t.d(e,"a",function(){return c});var n=t("S+eF"),r=t.n(n),o=t("fw5G"),i=t.n(o),a=t("CWYE"),s=t("sQ/U"),u=Object(a.a)("/api/onDemandHomeProgress.v1",{type:"rest"}),c=function getHomeProgress(e){var t="".concat(s.a.get().id,"~").concat(e),n=new i.a(t).addQueryParam("fields","modulesCompleted,modulesPassed");return r()(u.get(n.toString()))}},FUAI:function(module,e,t){"use strict";t.d(e,"a",function(){return a});var n=t("S+eF"),r=t.n(n),o=t("l490"),i=t("5ijc"),a=function loadVerificationDisplay(e,t){var n=t.authenticated,a=t.userId,s=t.courseId,u=t.s12nId,c=t.isCourseVerificationEnabled;if(e.getStore(i.a).hasLoaded())return r()();return n?Object(o.a)(a,s,c,u).then(function(t){e.dispatch("LOAD_VERIFICATION_DISPLAY",t)}):(e.dispatch("LOAD_VERIFICATION_DISPLAY",null),r()())}},JAlB:function(module,e,t){"use strict";var n=t("KMW/");e.a=function(e){return-1!==n.a.get("featureBlacklist","defaultDeadlines").indexOf(e)}},JSqB:function(module,e,t){"use strict";t.d(e,"a",function(){return a}),t.d(e,"b",function(){return s}),t.d(e,"c",function(){return u});var n=t("S+eF"),r=t.n(n),o=t("ROEb"),i=t("bZp4"),a=function loadHonorsUserPreferences(e,t){var n=t.authenticated;if(e.getStore("HonorsUserPreferencesStore").hasLoaded())return r()();return n?i.a.get(i.a.keyEnum.HONORS).then(function(t){e.dispatch("LOAD_HONORS_USER_PREFERENCES",t)}).fail(function(t){e.dispatch("LOAD_HONORS_USER_PREFERENCES",{})}):(e.dispatch("LOAD_HONORS_USER_PREFERENCES",{}),r()())},s=function setHonorsUserPreferences(e,t){var n=t.authenticated,o=t.updatedHonorsUserPreferences;return n?i.a.set(i.a.keyEnum.HONORS,o).then(function(){e.dispatch("LOAD_HONORS_USER_PREFERENCES",o)}):(e.dispatch("LOAD_HONORS_USER_PREFERENCES",o),r()())},u=function setLessonSkipped(e,t){var n=t.lessonId,r=t.skipped;e.dispatch("SET_LESSON_SKIPPED",{lessonId:n,skipped:r})}},JdaY:function(module,e,t){"use strict";t.r(e);var n=t("BJ98"),r=t.n(n),o=t("q1tI"),i=t.n(o),a=t("sQ/U"),s=t("EdUP"),u=t("kwmr"),c=t("+LJP"),d=t("NLLZ"),l=t("dAof"),f=t("Bgx3"),p=t("E4RX"),g=t("b+2U"),h=t("iTPM"),v=t("Re7p"),m=t("Shko"),S=t("+M5F"),y=t("Nher"),O=t("FUAI"),w=t("dgIx"),b=t("5Ujo"),E=t("JSqB"),I=t("fghW"),R=t("IG+7"),C=t("xPfO"),D=t("5ijc"),L=t("8c4I"),P=t("c2GL"),j=t("sjlm"),x=t("tPFS"),A=t("x+tN"),k=t("TOZ3"),F=t("8WNh"),M=function DataFetcherBody(e){var t=e.children;if(!t)return null;return i.a.cloneElement(t,{})},T=r()(Object(d.a)(function(){return!1}),Object(c.a)(function(e){return{courseSlug:e.params.courseSlug}}),Object(u.a)([C.a,R.a,P.a,I.a,D.a,j.a,L.a,x.a],function(e,t,n,r,o,i,a,s){return{s12n:r.getS12n(),course:t.getMetadata(),courseId:t.getCourseId(),isEnrolled:n.isEnrolled(),sessionId:e.getSessionId(),isEnrolledInSession:e.isEnrolled(),s12nStoreHasLoaded:r.hasLoaded(),courseStoreHasLoaded:t.hasLoaded(),sessionStoreHasLoaded:e.hasLoaded(),verificationStoreHasLoaded:o.hasLoaded(),courseMembershipStoreHasLoaded:n.hasLoaded(),computedModelStoreHasLoaded:a.hasLoaded(),courseIdentifiersHaveLoaded:t.haveCourseIdentifiersLoaded(),courseViewGradeStoreHasLoaded:i.hasLoaded(),progressStoreHasLoaded:s.hasLoaded()}}),Object(l.a)(function(e,t){var n=t.courseSlug;e.executeAction(f.a,n)}),Object(s.a)(function(e){var t;return e.courseIdentifiersHaveLoaded}),Object(s.a)(function(e){var t;return!!e.courseId},i.a.createElement(A.a,null)),Object(l.a)(function(e,t){var n=t.courseId;e.executeAction(m.a,n)}),Object(s.a)(function(e){var t;return e.courseMembershipStoreHasLoaded}),Object(s.a)(function(e){var t=e.isEnrolled;return a.a.isSuperuser()||t},i.a.createElement(A.a,null)),Object(l.a)(function(e,t){var n=t.courseSlug,r=t.courseId;e.executeAction(h.a,{courseSlug:n,courseId:r})}),Object(s.a)(function(e){var t;return e.computedModelStoreHasLoaded}),Object(l.a)(function(e,t){var n=t.courseId,r=t.courseSlug,o=a.a.get().id,i=a.a.isAuthenticatedUser();e.executeAction(g.a,{courseSlug:r}),e.executeAction(y.a,{courseId:n}),e.executeAction(w.a,{courseId:n,userId:o}),e.executeAction(E.a,{authenticated:i}),e.executeAction(v.a,{courseId:n,userId:o}),e.executeAction(p.a,{authenticated:i,courseId:n,userId:o})}),Object(s.a)(function(e){var t=e.s12nStoreHasLoaded,n=e.courseStoreHasLoaded,r=e.sessionStoreHasLoaded,o=e.courseViewGradeStoreHasLoaded,i=e.progressStoreHasLoaded;return t&&n&&r&&o&&i}),Object(l.a)(function(e,t){var n=t.courseId,r=t.course,o=t.s12n,i=t.sessionId,s=a.a.get().id,u=a.a.isAuthenticatedUser(),c=o&&o.getId(),d=r.isVerificationEnabled(),l=e.getStore("CourseStore");e.executeAction(S.a,{userId:s}),l.isCumulativeGradePolicy()&&e.executeAction(b.a,{userId:s,courseId:n}),e.executeAction(O.a,{authenticated:u,userId:s,courseId:n,isCourseVerificationEnabled:d,s12nId:c}),e.executeAction(w.b,{courseId:n,userId:s,sessionId:i})}),Object(s.a)(function(e){var t;return e.verificationStoreHasLoaded}))(M),N=function LegacyDataFetch(e){var t=e.children,n=e.isLegacyDataLoaded;return i.a.createElement("div",{className:"rc-LegacyDataFetch"},i.a.createElement(T,null,t),!n&&i.a.createElement(k.a,{height:512},i.a.createElement(F.a,null)))};e.default=r()(Object(u.a)([D.a],function(e){return{isLegacyDataLoaded:e.hasLoaded()}}))(N)},NBfQ:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("sQ/U"),i=t("fghW"),a=t("4San"),s=t("qujA");e.a=function(e,t){if(e.getStore(i.a).hasLoaded())return r()();var n;return Object(a.d)(t,o.a.get().id).then(function(e){var t=(n=e).elements,i=t&&t[0];return i&&o.a.isAuthenticatedUser()?Object(s.a)(i.id,!0):r()()}).then(function(t){return e.dispatch("LOAD_S12N",{rawS12ns:n,rawOwnership:t}),{rawS12ns:n,rawOwnership:t}})}},NLLZ:function(module,e,t){"use strict";t.d(e,"a",function(){return renderOrigamiIf});var n=t("VbXa"),r=t.n(n),o=t("q1tI"),i=t.n(o),a=t("17x9"),s=t.n(a),u=t("+MHB"),c=t("fw5G"),d=t.n(c),l=t("37kS"),f=t.n(l);function renderOrigamiIf(e){return function(t){var n=t.displayName||t.name,o=function(n){function RenderOrigamiIf(){for(var e,t=arguments.length,r=new Array(t),o=0;o<t;o++)r[o]=arguments[o];return(e=n.call.apply(n,[this].concat(r))||this).routerWillLeave=function(e){var t=u.a.renderedRegions.filter(function(e){return e.view.hasUnsavedModel}).map(function(e){return e.view.hasUnsavedModel()}).reduce(function(e,t){return e||t},!1);if(!t)return!0;return!(!t||!window.confirm(f()("There are unsaved changes that will be lost if you reload or leave this page.")))},e}r()(RenderOrigamiIf,n);var o=RenderOrigamiIf.prototype;return o.componentDidMount=function componentDidMount(){var e=this,t=this.context.router,n=this.props.route,r=function navigate(r){var o=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},i=new d.a(r),a=new d.a(window.location.toString());if(a.path()===i.path()&&a.query()===i.query())return;!0===o||o&&!0===o.trigger?o&&o.replace?t.replace(r):t.push(r):o&&o.replace?window.history.replaceState(null,null,r):window.history.pushState(null,null,r),n&&t.setRouteLeaveHook(n,e.routerWillLeave)};u.a.router.navigate=r,u.a.router.navigateTo=r},o.componentWillReceiveProps=function componentWillReceiveProps(t,n){e(this.props)&&!e(t)&&u.a.trigger("close")},o.render=function render(){return e(this.props)?i.a.createElement("div",null,i.a.createElement("div",{"data-js":"origami"}),i.a.createElement(t,this.props)):i.a.createElement(t,this.props)},RenderOrigamiIf}(i.a.Component);return o.displayName="RenderOrigamiIf(".concat(n,")"),o.propTypes={children:s.a.node,route:s.a.object},o.contextTypes={router:s.a.object},o}}},NO4R:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("BYIE"),i=t("qgMw"),a=t("lqQ6");e.a=function(e,t){if(e.getStore(i.a).hasLoaded())return r()();if(!t)return r.a.reject(new a.a("courseId must be provided."));return Object(o.a)(t).then(function(t){e.dispatch("LOAD_COURSE_SCHEDULE",t)})}},PVyl:function(module,e,t){"use strict";t.r(e);var n=t("qlEU"),r=t.n(n),o=t("KgRe"),i=t("cySH");e.default=function(e){return r()(i.a.build(o.a.prototype.resourceName,e))}},Re7p:function(module,e,t){"use strict";t.d(e,"a",function(){return u});var n=t("lSNA"),r=t.n(n),o=t("S+eF"),i=t.n(o),a=t("yF8l");function ownKeys(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function _objectSpread(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?ownKeys(Object(n),!0).forEach(function(t){r()(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):ownKeys(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var s={showHidden:!0,fields:["courseId","grade"],includes:{vcMembership:{fields:["certificateCode","grade","grantedAt"]},course:{fields:[]}}},u=function loadCertificateData(e,t){var n=t.courseId,r=t.userId,o;if(e.getStore("CertificateStore").hasLoaded())return i()();return(o=r?Object(a.a)(_objectSpread(_objectSpread({id:"".concat(r,"~").concat(n)},s),{},{rawData:!0})).then(function(t){e.dispatch("LOAD_MEMBERSHIPS",t)}):i()().then(function(){e.dispatch("LOAD_MEMBERSHIPS",null)})).done(),o}},TFmq:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("sQ/U"),i=t("tPFS"),a=t("D87r");e.a=function(e,t){if(e.getStore(i.a).hasLoaded())return r()();return o.a.isAuthenticatedUser()?Object(a.a)(t).then(function(t){t.elements&&t.elements.length&&e.dispatch("LOAD_HOME_PROGRESS",t.elements[0])}).fail(function(){e.dispatch("LOAD_HOME_PROGRESS",{modulesCompleted:[],modulesPassed:[]})}):(e.dispatch("LOAD_HOME_PROGRESS",{modulesCompleted:[],modulesPassed:[]}),r()())}},Vu1r:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("fw5G"),i=t.n(o),a=t("DnuM"),s=Object(a.a)("/api/onDemandCoursePresentGrades.v1",{type:"rest"}),u=function getPresentGrade(e){var t=e.userId,n=e.courseId,o=new i.a("/".concat(t,"~").concat(n)).addQueryParam("fields","grade,relevantItems,passingStateForecast");return r()(s.get(o.toString()))};e.a={getPresentGrade:u}},Y3VV:function(module,e,t){"use strict";var n=t("lSNA"),r=t.n(n),o=t("VbXa"),i=t.n(o),a=t("17x9"),s=t.n(a),u=t("q1tI"),c=t.n(u);function ownKeys(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function _objectSpread(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?ownKeys(Object(n),!0).forEach(function(t){r()(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):ownKeys(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var d=function(e){function ContextProvider(t){var n;n=e.call(this,t)||this;var r={};return Object.keys(t.customContext).forEach(function(e){r[e]=s.a.any}),ContextProvider.childContextTypes=_objectSpread(_objectSpread({},ContextProvider.childContextTypes),r),n}i()(ContextProvider,e);var t=ContextProvider.prototype;return t.getChildContext=function getChildContext(){return Object.assign({},this.props.customContext)},t.render=function render(){return c.a.cloneElement(this.props.children,this.props)},ContextProvider}(c.a.Component);d.displayName=d,d.childContextTypes={},e.a=d},"b+2U":function(module,e,t){"use strict";t.d(e,"a",function(){return a}),t.d(e,"b",function(){return s});var n=t("S+eF"),r=t.n(n),o=t("hvww"),i=t("xPfO"),a=function getCurrentSession(e,t){var n=t.courseSlug;if(e.getStore(i.a).hasLoaded())return r()();return o.d(n).then(function(t){e.dispatch("LOAD_SESSION",t||null)}).fail(function(e){throw e})},s=function updateEnrollableAndFollowingSessions(e,t){var n=t.courseId,a=t.currentSessionId;if(e.getStore(i.a).hasLoaded())return r()();return o.e(n,a).then(function(t){(t.getUpcomingSession()||t.getFollowingSession())&&e.dispatch("LOAD_UPCOMING_AND_FOLLOWING_SESSIONS",{upcomingSession:t.getUpcomingSession(),followingSession:t.getFollowingSession()})}).fail(function(e){throw e})},u=function getAllSessions(e,t){if(e.getStore(i.a).hasLoaded())return r()();return o.c(t).then(function(t){e.dispatch("LOAD_ALL_SESSIONS",t)})}},dgIx:function(module,e,t){"use strict";t.d(e,"a",function(){return c}),t.d(e,"b",function(){return d});var n=t("S+eF"),r=t.n(n),o=t("F/us"),i=t.n(o),a=t("yFL5"),s=t("6p3O"),u=t("Aw3H"),c=function loadUserGroupsForCourse(e,t){var n=t.courseId,o=t.userId;if(e.getStore("GroupSettingStore").hasLoaded())return r()();return a.a.myCourseGroupsWithSettings(o,n).then(function(t){var n=i()(t.linked["groupSettings.v1"]).map(function(e){return new u.a(e)}),r=t.linked["groups.v1"].map(function(e){return new s.a(e)}),o=t.elements;e.dispatch("LOADED_COURSE_GROUPS",{groups:r,groupSettings:n,groupMemberships:o})}).fail(function(t){e.dispatch("LOADED_COURSE_GROUPS",{})})},d=function loadUserSessionGroupForCourse(e,t){var n=t.courseId,o=t.userId,i=t.sessionId;if(e.getStore("GroupSettingStore").hasSessionGroupLoaded())return r()();return a.a.getCourseSessionGroup(o,n,i).then(function(t){var n=t.elements[0];e.dispatch("LOADED_SESSION_GROUP",{sessionGroup:n})}).fail(function(t){e.dispatch("LOADED_SESSION_GROUP",{})})}},iTPM:function(module,e,t){"use strict";t.d(e,"a",function(){return l});var n=t("S+eF"),r=t.n(n),o=t("NBfQ"),i=t("sroZ"),a=t("NO4R"),s=t("hw75"),u=t("wbHF"),c=t("TFmq"),d=t("8c4I"),l=function loadComputedModels(e,t){var n=t.courseSlug,l=t.courseId;if(e.getStore(d.a).hasLoaded())return r()();return r.a.all([Object(i.a)(e),Object(o.a)(e,l),Object(s.a)(e,n),Object(a.a)(e,l),Object(u.a)(e,l),Object(c.a)(e,l)]).then(function(){e.dispatch("LOAD_COMPUTED_MODELS")})}},j4LP:function(module,e,t){"use strict";t.d(e,"a",function(){return createMountNode}),t.d(e,"b",function(){return runTrackingRouter});var n=t("1mBx"),r=t.n(n),o=t("q1tI"),i=t.n(o),a=t("i8i4"),s=t.n(a),u=t("tta6"),c=t("3Hqd");function createMountNode(e){var t=document.getElementById(e);return t||((t=document.createElement("div")).id=e,document.body.appendChild(t)),t}function runTrackingRouter(e,t){return Object(u.a)({routes:e}).then(function(e){return s.a.render(i.a.createElement(r.a,e),t),e.router})}var d,l={createMountNode:createMountNode,runTrackingRouter:runTrackingRouter}},l490:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("uyIS");e.a=function(e,t,n,i){if(n&&e){var a=r.a.all([Object(o.a)(e,t,!0)]).spread(function(e){var t;return{isProductVerificationEnabled:n,productOwnership:e,s12nId:i}},function(){return null});return a.done(),a}var s=r()(null);return s.done(),s}},sArp:function(module,e,t){"use strict";t.d(e,"a",function(){return u});var n=t("S+eF"),r=t.n(n),o=t("fw5G"),i=t.n(o),a=t("CWYE"),s=Object(a.a)("/api/onDemandCourseSchedules.v1"),u=function getCourseSchedule(e){var t=new i.a(e).addQueryParam("fields","defaultSchedule");return r()(s.get(t.toString()))}},sroZ:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("DnuM"),i=t("IG+7");e.a=function(e){var t=Object(o.a)("/api/domains.v1",{type:"rest"});if(void 0!==e.getStore(i.a).domains)return r()();return r()(t.get("?fields=id,name")).then(function(t){e.dispatch("LOAD_DOMAINS",t.elements)})}},tDXo:function(module,e,t){"use strict";var n=t("lSNA"),r=t.n(n),o=t("cDf5"),i=t.n(o),a=t("oYk5"),s=t.n(a),u=t("S+eF"),c=t.n(u),d=t("q1tI"),l=t.n(d),f=t("i8i4"),p=t.n(f),g=t("F/us"),h=t.n(g),v=t("rC9M"),m=t.n(v),S=t("Y3VV"),y=t("7MHW"),O=t("ROEb"),w=t("r9wp"),b=t("welz"),E=t.n(b),require;function ownKeys(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function _objectSpread(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?ownKeys(Object(n),!0).forEach(function(t){r()(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):ownKeys(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var I="backbone_region",R="react_region";E.a.View.prototype._super=function(e){return this.constructor.__super__[e].apply(this,h.a.rest(arguments))};var C=null,D={current:null,body:"#origami",views:{},openDepth:0,beforeUnloadMessage:"There are unsaved changes that will be lost if you reload or leave this page.",regionHasUnsavedModel:function regionHasUnsavedModel(e){if(!e)return!1;var t;if(!(!e.view||!e.view.hasUnsavedModel)&&e.view.hasUnsavedModel())return!0;for(var n in e.regions){var r=e.regions[n];if(D.regionHasUnsavedModel(r))return!0}return!1}},L=E.a.Router.extend({initialize:function initialize(){var e=this;this.routePrefix="",this.visitedFragments=[],this.$confirmNavigation=s()(m()({})),s()("body").append(this.$confirmNavigation),s()(window).on("beforeunload.origami",function(){if(D.regionHasUnsavedModel(D.current))return D.beforeUnloadMessage})},setupLinks:function setupLinks(e,t){var n=this;this.routePrefix=e,this.linkSelector=t,s()("body").on("click","a"+this.linkSelector,function(e){e.ctrlKey||e.metaKey||(e.preventDefault(),n.navigateTo(s()(e.currentTarget).attr("href"),!0))})},addFragmentToVisited:function addFragmentToVisited(e,t){if("object"===i()(t)&&t.replace)return;this.visitedFragments?this.visitedFragments.push(e):this.visitedFragments=[e]},navigateTo:function navigateTo(e,t){e||O.a.error("router.navigateTo received no href"),this.navigate(e.replace(this.routePrefix,""),t)},navigate:function navigate(e,t){if(D.current&&D.regionHasUnsavedModel(D.current)){var n,r;s()(".confirm-navigation-message").text(D.beforeUnloadMessage),s()(".confirm-navigation-leave").one("click",function(n){this.addFragmentToVisited(e),E.a.Router.prototype.navigate.call(n.currentTarget,e,t)}.bind(this));var o={"overlay.class":"coursera-overlay"};Object(w.default)(".confirm-navigation",o).open()}else this.addFragmentToVisited(e,t),E.a.Router.prototype.navigate.call(this,e,t)},addRoutes:function addRoutes(e,t){var n=this;for(var r in e){var o=e[r];t&&(o=t(o)),n.route(r,r,o),n.route(r+"/",r+"/",function(){var e,t=E.a.history.getFragment().replace(/\/(?:\?.*)*$/,"");n.navigate(t,{trigger:!0,replace:!0})})}},start:function start(e){var t;E.a.history.start({pushState:!0,hashChange:!1,root:e})||this.trigger("error",404)},unsavedWarningModal:function unsavedWarningModal(e){if(D.current&&D.regionHasUnsavedModel(D.current)){var t={"overlay.class":"coursera-overlay"},n;s()(".confirm-navigation-message").text(D.beforeUnloadMessage);var r=Object(w.default)(".confirm-navigation",t);r.once("action",function(t){e(t)}),r.open()}else e(!0)}}),P=function region(e,t){var n=this,r=t||{};this.type=e,this.regions=r.regions||{},this.initialize=r.initialize||{},this.to=r.to,this.options={},this.props=t.props||{},this.context=t.context||{},this.fluxibleContext=t.fluxibleContext,this.reactRouter=t.reactRouter,this.id=r.id||e.prototype.name,this.name=e.prototype.name||r.name,this.view=null,this.regions=s.a.extend({},e.prototype.subregions,r.regions||{})};P.prototype.close=function(e){var t=this;h.a.each(t.regions,function(n,r){delete t.regions[r],n&&n.close(e)}),t.view.trigger("view:closed"),e||t.view.remove(),t.view.undelegateEvents(),t.view.off("view")},P.prototype.closing=function(){this.view.trigger("view:closing"),delete D.views[this.view.cid]},P.prototype.render=function(e){var t=this,n=h.a.compact(h.a.values(t.regions));return h.a.each(n,function(e){e.render(),e.parent=t}),this.type.prototype.isReactComponent?(this.kind=R,this.renderReactComponent()):(this.kind=I,this.renderBackboneView()),h.a.each(n,function(e){var n=e.view&&e.view.el,r=e.to||e.view&&e.view.to;h.a.isFunction(r)&&(r=r()),r&&!s.a.contains(document,n)&&(t.view.$(r).append(n),e.appended())}),e&&t.view.on("view:appended",function(){e.trigger("region:appended",{name:t.name})}),t},P.prototype.renderBackboneView=function(){this.view=new this.type(this.initialize),this.view.region=this,this.view.render(this.render)},P.prototype.renderReactComponent=function(){var e=this.context.fluxibleContext,t=this.context.router,n=Object.assign({},e?_objectSpread({fluxibleContext:e},e):{},t?{router:t}:{}),r=h.a.compose(y.a)(this.type);this.view=new E.a.View,this.view.region=this;var o=l.a.createElement(S.a,{customContext:n},l.a.createElement(r,this.props));this.view.remove=function(){return p.a.unmountComponentAtNode(this.el),E.a.View.prototype.remove.call(this),this},this.view.on("view:appended",function(){p.a.render(o,this.el)})},P.prototype.appended=function(){var e=this;e.view&&setTimeout(function(){e.view.trigger("view:appended")},0)},P.prototype.append=function(e,t){var n=this,r=c.a.defer();this.regions||(this.regions={});var o=function renderAndAppend(o){for(var i=e,a=0,u;n.regions[i];)i+=++a;if(n.regions[i]=o,o.render(C),u=t.to?h.a.isElement(t.to)?n.view.$(t.to):t.to:n.view.$el,h()(t).has("position")){var c=u.children().eq(t.position);c.length>0?s()(c[0]).before(o.view.el):u.append(o.view.el)}else u.append(o.view.el);o.appended(),C.renderedRegions.push(o),C.trigger("region:rendered",{name:o}),r.resolve(o)};return void 0!==e&&("string"==typeof e?C.region.fetch(e,t,o):o(new P(e,t))),r.promise},P.prototype.is=function(e){return e.type==this.type&&e.id==this.id},P.prototype.replace=function(e,t){this.view.$el.replaceWith(e.view.el),t&&t()},P.prototype.merge=function(e,t){var n=this;n.is(e)?h.a.each(e.regions,function(e,r){var o=n.regions[r];o&&e?(o.view.trigger("view:merging",e.initialize),o.merge(e,t),o.view.trigger("view:merged",e.initialize)):o&&!e&&(o.closing(),o.parent&&delete o.parent.regions[o.name],o.close())}):n.name==e.name&&(n.closing(),e.render(t),n.replace(e,function(){n.parent&&(n.parent.regions[e.name]=e,e.parent=n.parent,n.parent.appended()),n.close(),e.appended()}))};var j=function Origami(e,n){if(C)return C;this.renderedRegions=[],C=this,D.body=e,D.current=n,s()(window).on("resize",function(){for(var e in D.views)D.views[e].trigger("view:resize")}),this.region={make:function make(e,t){return new P(e,t)},fetch:function fetch(e,n,r){var o=h.a.uniqueId();C.trigger("region:fetching",{name:e,uid:o});var i=function returnRegion(t){C.trigger("region:fetched",{name:e,uid:o}),r(new P(t,n))},module=n.module,a=n.async;if(module&&module.lazy||module&&a)module(i);else if(module)i(module);else{var s,u;u=(null===(s=jest)||void 0===s?void 0:s.requireActual)?function req(e,n){var r;return n(t("AbQT")(e[0]))}:void 0,C.trigger("deprecatedUsage",{type:"asyncRequire",name:e}),u([e],i)}},getSingleton:function getSingleton(){return C},append:function append(e,t,n,r){e||(e={regions:{}});var o=function renderAndAppend(n){for(var o=t,i=0;e.regions[o];)o+=++i;return e.regions[o]=n,n.render(C),r&&r(n),n.appended(),C.renderedRegions.push(n),C.trigger("region:rendered",{name:n}),n};if(void 0!==t){if("string"!=typeof t)return o(new P(t,n));C.region.fetch(t,n,o)}},open:function open(e,t){var n=e,r={};0===D.openDepth&&C.trigger("open"),D.openDepth++;var o=function onFetch(e){var n=[];h.a.each(e.regions,function(t,r){if(!t)return;var o=c.a.defer();C.region.open(t,function(t){t.kind==I&&r!=t.name&&O.a.error("Region configured with name "+r+" but has name "+t.name),e.regions[r]=t,o.resolve(t)}),n.push(o.promise)}),c.a.allSettled(n).then(function(){if(t)return t(e);if(D.current){if(D.current.is(e))return void D.current.merge(e,C);if(D.current.name==e.name)return e.render(C),void D.current.replace(e,function(){D.current.close(),D.current=e,D.current.appended()});D.current.close(!0)}C.dontRender||(D.current=e,s()(D.body).html(D.current.render(C).view.$el),D.current.appended())}).done(function(){--D.openDepth<=0&&(C.on("close",function(){C.renderedRegions=[],D.current&&(D.current.view.once("view:closed",function(){D.current=void 0}),D.current.close.call(D.current))}),C.trigger("opened"))})};P.prototype.isPrototypeOf(e)?o(e):(h.a.isObject(e)&&(r=e[n=h.a.keys(e)[0]]),C.region.fetch(n,r,o))}},this.interruptRender=function(){this.dontRender=!0}.bind(this),this.cancelInterrupt=function(){this.dontRender=!1}.bind(this),this.router=new L,this.store={},this.tracker=[]};h.a.extend(j.prototype,E.a.Events),e.a=j},wDa7:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("CWYE"),i=t("fw5G"),a=t.n(i),s=t("sQ/U"),u=Object(o.a)("/api/onDemandDeadlineSettings.v1",{type:"rest"}),c={getStartTime:function getStartTime(e){var t=(new a.a).addQueryParam("q","byUserAndCourse").addQueryParam("userId",s.a.get().id).addQueryParam("courseId",e).toString();return r()(u.get(t)).fail(function(e){console.error(e)})},sendStartTime:function sendStartTime(e,t){var n={data:{userId:s.a.get().id,courseId:t,start:Date.now(),isEnabled:e}};return r()(u.post("",n))},disableDeadlines:function disableDeadlines(e){return c.sendStartTime(!1,e)},getResetPreview:function getResetPreview(e,t){var n=(new a.a).addQueryParam("q","extendPreview").addQueryParam("userId",s.a.get().id).addQueryParam("courseId",e).addQueryParam("extendedAt",Date.now()).toString();r()(u.get(n)).then(t).fail(function(e){console.error(e)}).done()},resetDeadlines:function resetDeadlines(e){var t={data:{userId:s.a.get().id,courseId:e,extendedAt:Date.now()}},n=(new a.a).addQueryParam("action","extend").toString();return r()(u.post(n,t))}};e.a=c;var d=c.getStartTime,l=c.sendStartTime,f=c.disableDeadlines,p=c.getResetPreview,g=c.resetDeadlines},wbHF:function(module,e,t){"use strict";var n=t("S+eF"),r=t.n(n),o=t("fw5G"),i=t.n(o),a=t("CWYE");e.a=function(e,t){var n=Object(a.a)("/api/onDemandReferences.v1",{type:"rest"}),o=(new i.a).addQueryParam("courseId",t).addQueryParam("q","courseListed").addQueryParam("fields","name,shortId,slug,content").addQueryParam("includes","assets");return r()(n.get(o.toString())).then(function(t){e.dispatch("LOAD_REFERENCES_LIST",t.elements)})}},yF8l:function(module,e,t){"use strict";var n=t("F/us"),r=t.n(n),o=t("ywP/"),i=t.n(o),a=t("f5V2"),s=t("9CUK"),u=t("Toj+");e.a=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};return i()(e).then(function(t){if(t.linked&&t.linked["onDemandSessions.v1"]&&t.linked["onDemandSessionMemberships.v1"]){var n=r()(t.linked["onDemandSessions.v1"]).groupBy("courseId"),o=r()(t.linked["onDemandSessionMemberships.v1"]).groupBy("sessionId"),i=Object.keys(o);t.elements.forEach(function(e){var t=n[e.courseId]||[];if(t.length){var r=t.filter(function(e){return i.indexOf(e.id)>=0});if(r.length){var a=new u.b(r).getLastSession();e.onDemandSessionId=a.id,e.onDemandSessionMemberships=r.map(function(e){return o[e.id]})}}})}if(t.linked&&t.linked["v1Details.v1"]&&(t.linked["courses.v1"]=r()(t.linked["courses.v1"]).map(function(e){if("v1.session"===e.courseType||"v1.capstone"===e.courseType){e.v1Details=e.id;var n=r()(t.linked["v1Sessions.v1"]).reduce(function(t,n){return n.courseId===e.id&&t.push(n.id.toString()),t},[]);e.v1Sessions=n}return e})),t.linked&&t.linked["v2Details.v1"]&&(t.linked["courses.v1"]=r()(t.linked["courses.v1"]).map(function(e){return"v2.ondemand"===e.courseType&&(e.v2Details=r()(t.linked["v2Details.v1"]).findWhere({id:e.id})),e})),t.linked&&t.linked["vcMemberships.v1"]){var c=r()(t.linked["vcMemberships.v1"]).pluck("id");t.elements=r()(t.elements).map(function(e){return r()(c).contains(e.id)&&(e.vcMembershipId=e.id),e})}if(t.linked&&t.linked["courses.v1"]){var d=r()(t.linked["courses.v1"]).pluck("id");t.elements=r()(t.elements).chain().filter(function(e){return r()(d).contains(e.courseId)}).compact().value()}if(t.linked&&t.linked["signatureTrackProfiles.v1"]&&r()(t.elements).each(function(e){e.signatureTrackProfile=e.userId}),e.rawData)return t;if(e.withPaging)return{elements:Object(a.a)(s.a.prototype.resourceName,t),paging:t.paging};return Object(a.a)(s.a.prototype.resourceName,t)}).fail(function(t){if(e.rawData)return null;return new s.a})}},"ywP/":function(module,exports,e){var t=function(e){return e.default?e.default:e},n=e("8kE/").default,r=t(e("PVyl"));module.exports=n(r)}}]);
//# sourceMappingURL=7.a7cfb2184fc0c68eb3ef.js.map